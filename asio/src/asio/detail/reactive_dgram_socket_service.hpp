//
// dgram_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind.hpp>
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/completion_context.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
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

  // Constructor.
  reactive_dgram_socket_service(Demuxer& d)
    : demuxer_(d),
      reactor_(d.get_service(service_factory<Reactor>()))
  {
  }

  // Return a null dgram socket implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // Create a new dgram socket implementation.
  template <typename Address>
  void create(impl_type& impl, const Address& address)
  {
    socket_holder sock(socket_ops::socket(address.family(), SOCK_DGRAM,
          IPPROTO_UDP));
    if (sock.get() == invalid_socket)
      boost::throw_exception(socket_error(socket_ops::get_error()));

    int reuse = 1;
    socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
        sizeof(reuse));

    if (socket_ops::bind(sock.get(), address.native_address(),
          address.native_size()) == socket_error_retval)
      boost::throw_exception(socket_error(socket_ops::get_error()));

    impl = sock.release();
  }

  // Destroy a dgram socket implementation.
  void destroy(impl_type& impl)
  {
    if (impl != null())
    {
      reactor_.close_descriptor(impl);
      socket_ops::close(impl);
      impl = null();
    }
  }

  // Send a datagram to the specified address. Returns the number of bytes
  // sent. Throws a socket_error exception on failure.
  template <typename Address>
  size_t sendto(impl_type& impl, const void* data, size_t length,
      const Address& destination)
  {
    int bytes_sent = socket_ops::sendto(impl, data, length, 0,
        destination.native_address(), destination.native_size());
    if (bytes_sent < 0)
      boost::throw_exception(socket_error(socket_ops::get_error()));
    return bytes_sent;
  }

  template <typename Address, typename Handler>
  class sendto_handler
  {
  public:
    sendto_handler(impl_type impl, Demuxer& demuxer, const void* data,
        size_t length, const Address& address, Handler handler,
        completion_context& context)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        length_(length),
        destination_(address),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      int bytes = socket_ops::sendto(impl_, data_, length_, 0,
          destination_.native_address(), destination_.native_size());
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_.operation_completed(boost::bind(handler_, error, bytes),
          context_);
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(boost::bind(handler_, error, 0), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    const void* data_;
    size_t length_;
    Address destination_;
    Handler handler_;
    completion_context& context_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Address, typename Handler>
  void async_sendto(impl_type& impl, const void* data, size_t length,
      const Address& destination, Handler handler,
      completion_context& context)
  {
    demuxer_.operation_started();
    reactor_.start_write_op(impl, sendto_handler<Address, Handler>(impl,
          demuxer_, data, length, destination, handler, context));
  }

  // Receive a datagram with the address of the sender. Returns the number of
  // bytes received. Throws a socket_error exception on failure.
  template <typename Address>
  size_t recvfrom(impl_type& impl, void* data, size_t max_length,
      Address& sender_address)
  {
    socket_addr_len_type addr_len = sender_address.native_size();
    int bytes_recvd = socket_ops::recvfrom(impl, data, max_length, 0,
        sender_address.native_address(), &addr_len);
    if (bytes_recvd < 0)
      boost::throw_exception(socket_error(socket_ops::get_error()));
    sender_address.native_size(addr_len);
    return bytes_recvd;
  }

  template <typename Address, typename Handler>
  class recvfrom_handler
  {
  public:
    recvfrom_handler(impl_type impl, Demuxer& demuxer, void* data,
        size_t max_length, Address& address, Handler handler,
        completion_context& context)
      : impl_(impl),
        demuxer_(demuxer),
        data_(data),
        max_length_(max_length),
        sender_address_(address),
        handler_(handler),
        context_(context)
    {
    }

    void do_operation()
    {
      socket_addr_len_type addr_len = sender_address_.native_size();
      int bytes = socket_ops::recvfrom(impl_, data_, max_length_, 0,
          sender_address_.native_address(), &addr_len);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      sender_address_.native_size(addr_len);
      demuxer_.operation_completed(boost::bind(handler_, error, bytes),
          context_);
    }

    void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_.operation_completed(boost::bind(handler_, error, 0), context_);
    }

  private:
    impl_type impl_;
    Demuxer& demuxer_;
    void* data_;
    size_t max_length_;
    Address& sender_address_;
    Handler handler_;
    completion_context& context_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_address obejct must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Address, typename Handler>
  void async_recvfrom(impl_type& impl, void* data, size_t max_length,
      Address& sender_address, Handler handler, completion_context& context)
  {
    demuxer_.operation_started();
    reactor_.start_read_op(impl, recvfrom_handler<Address, Handler>(impl,
          demuxer_, data, max_length, sender_address, handler, context));
  }

private:
  // The demuxer used for delivering completion notifications.
  Demuxer& demuxer_;

  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP
