//
// reactive_dgram_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename Demuxer, typename Reactor>
class reactive_dgram_socket_service
{
public:
  // The native type of the dgram socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // Return a null dgram socket implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // Constructor.
  reactive_dgram_socket_service(Demuxer& d)
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

  // Open a new dgram socket implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    if (protocol.type() != SOCK_DGRAM)
    {
      error_handler(socket_error(socket_error::invalid_argument));
      return;
    }

    socket_holder sock(socket_ops::socket(protocol.family(), protocol.type(),
          protocol.protocol()));
    if (sock.get() == invalid_socket)
      error_handler(socket_error(socket_ops::get_error()));
    else
      impl = sock.release();
  }

  // Bind the dgram socket to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, endpoint.native_data(),
          endpoint.native_size()) == socket_error_retval)
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Destroy a dgram socket implementation.
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

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = endpoint.native_size();
    if (socket_ops::getsockname(impl, endpoint.native_data(), &addr_len))
      error_handler(socket_error(socket_ops::get_error()));
    endpoint.native_size(addr_len);
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename Endpoint, typename Error_Handler>
  size_t sendto(impl_type& impl, const void* data, size_t length,
      const Endpoint& destination, Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::sendto(impl, data, length, 0,
        destination.native_data(), destination.native_size());
    if (bytes_sent < 0)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return 0;
    }
    return bytes_sent;
  }

  template <typename Endpoint, typename Handler>
  class sendto_handler
  {
  public:
    sendto_handler(impl_type impl, Demuxer& demuxer, const void* data,
        size_t length, const Endpoint& endpoint, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        length_(length),
        destination_(endpoint),
        handler_(handler)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::sendto(impl_, data_, length_, 0,
          destination_.native_data(), destination_.native_size());
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    const void* data_;
    size_t length_;
    Endpoint destination_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Endpoint, typename Handler>
  void async_sendto(impl_type& impl, const void* data, size_t length,
      const Endpoint& destination, Handler handler)
  {
    if (impl == null())
    {
      socket_error error(socket_error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_write_op(impl, sendto_handler<Endpoint, Handler>(
            impl, demuxer_, data, length, destination, handler));
    }
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename Endpoint, typename Error_Handler>
  size_t recvfrom(impl_type& impl, void* data, size_t max_length,
      Endpoint& sender_endpoint, Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = sender_endpoint.native_size();
    int bytes_recvd = socket_ops::recvfrom(impl, data, max_length, 0,
        sender_endpoint.native_data(), &addr_len);
    if (bytes_recvd < 0)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return 0;
    }

    sender_endpoint.native_size(addr_len);

    return bytes_recvd;
  }

  template <typename Endpoint, typename Handler>
  class recvfrom_handler
  {
  public:
    recvfrom_handler(impl_type impl, Demuxer& demuxer, void* data,
        size_t max_length, Endpoint& endpoint, Handler handler)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        max_length_(max_length),
        sender_endpoint_(endpoint),
        handler_(handler)
    {
    }

    void do_operation()
    {
      socket_addr_len_type addr_len = sender_endpoint_.native_size();
      int bytes = socket_ops::recvfrom(impl_, data_, max_length_, 0,
          sender_endpoint_.native_data(), &addr_len);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      sender_endpoint_.native_size(addr_len);
      demuxer_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      demuxer_.work_finished();
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.post(bind_handler(handler_, error, 0));
      demuxer_.work_finished();
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    void* data_;
    size_t max_length_;
    Endpoint& sender_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Endpoint, typename Handler>
  void async_recvfrom(impl_type& impl, void* data, size_t max_length,
      Endpoint& sender_endpoint, Handler handler)
  {
    if (impl == null())
    {
      socket_error error(socket_error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error, 0));
    }
    else
    {
      demuxer_.work_started();
      reactor_.start_read_op(impl, recvfrom_handler<Endpoint, Handler>(
            impl, demuxer_, data, max_length, sender_endpoint, handler));
    }
  }

private:
  // The demuxer used for dispatching handlers.
  Demuxer& demuxer_;

  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP
