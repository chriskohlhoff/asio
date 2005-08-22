//
// reactive_stream_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"
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

  // The demuxer type for this service.
  typedef Demuxer demuxer_type;

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Open a new stream socket implementation.
  void open(impl_type& impl, impl_type new_impl)
  {
    impl = new_impl;
  }

  // Destroy a stream socket implementation.
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      reactor_.close_descriptor(impl, socket_ops::close);
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
  void get_option(const impl_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getsockname(impl, endpoint.native_data(), &addr_len))
      error_handler(asio::error(socket_ops::get_error()));
    endpoint.native_size(addr_len);
  }

  // Get the remote endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getpeername(impl, endpoint.native_data(), &addr_len))
      error_handler(asio::error(socket_ops::get_error()));
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
  class send_handler
  {
  public:
    send_handler(impl_type impl, Demuxer& demuxer, const void* data,
        size_t length, socket_base::message_flags flags, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        length_(length),
        flags_(flags),
        handler_(handler)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::send(impl_, data_, length_, flags_);
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    const void* data_;
    size_t length_;
    socket_base::message_flags flags_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Handler>
  void async_send(impl_type& impl, const void* data, size_t length,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_write_op(impl,
          send_handler<Handler>(impl, demuxer_, data, length, flags, handler));
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
  class receive_handler
  {
  public:
    receive_handler(impl_type impl, Demuxer& demuxer, void* data,
        size_t max_length, socket_base::message_flags flags, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        max_length_(max_length),
        flags_(flags),
        handler_(handler)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::recv(impl_, data_, max_length_, flags_);
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      asio::error error(asio::error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    void* data_;
    size_t max_length_;
    socket_base::message_flags flags_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Handler>
  void async_receive(impl_type& impl, void* data, size_t max_length,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      if (flags & socket_base::message_out_of_band)
      {
        reactor_.start_except_op(impl,
            receive_handler<Handler>(impl, demuxer_,
              data, max_length, flags, handler));
      }
      else
      {
        reactor_.start_read_op(impl,
            receive_handler<Handler>(impl, demuxer_,
              data, max_length, flags, handler));
      }
    }
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
  // The demuxer used for dispatching handlers.
  Demuxer& demuxer_;

  // The reactor that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_STREAM_SOCKET_SERVICE_HPP
