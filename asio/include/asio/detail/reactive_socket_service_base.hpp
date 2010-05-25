//
// detail/reactive_socket_service_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_BASE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/utility/addressof.hpp>
#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/null_buffers_op.hpp"
#include "asio/detail/reactor.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_recv_op.hpp"
#include "asio/detail/socket_send_op.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class reactive_socket_service_base
{
public:
  // The native type of a socket.
  typedef socket_type native_type;

  // The implementation type of the socket.
  struct base_implementation_type
  {
    // The native socket representation.
    socket_type socket_;

    // The current state of the socket.
    socket_ops::state_type state_;

    // Per-descriptor data used by the reactor.
    reactor::per_descriptor_data reactor_data_;
  };

  // Constructor.
  reactive_socket_service_base(asio::io_service& io_service)
    : reactor_(use_service<reactor>(io_service))
  {
    reactor_.init_task();
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  // Construct a new socket implementation.
  void construct(base_implementation_type& impl)
  {
    impl.socket_ = invalid_socket;
    impl.state_ = 0;
  }

  // Destroy a socket implementation.
  void destroy(base_implementation_type& impl)
  {
    if (impl.socket_ != invalid_socket)
    {
      reactor_.close_descriptor(impl.socket_, impl.reactor_data_);

      asio::error_code ignored_ec;
      socket_ops::close(impl.socket_, impl.state_, true, ignored_ec);
    }
  }

  // Determine whether the socket is open.
  bool is_open(const base_implementation_type& impl) const
  {
    return impl.socket_ != invalid_socket;
  }

  // Destroy a socket implementation.
  asio::error_code close(base_implementation_type& impl,
      asio::error_code& ec)
  {
    if (is_open(impl))
      reactor_.close_descriptor(impl.socket_, impl.reactor_data_);

    if (socket_ops::close(impl.socket_, impl.state_, true, ec) == 0)
      construct(impl);

    return ec;
  }

  // Get the native socket representation.
  native_type native(base_implementation_type& impl)
  {
    return impl.socket_;
  }

  // Cancel all operations associated with the socket.
  asio::error_code cancel(base_implementation_type& impl,
      asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return ec;
    }

    reactor_.cancel_ops(impl.socket_, impl.reactor_data_);
    ec = asio::error_code();
    return ec;
  }

  // Determine whether the socket is at the out-of-band data mark.
  bool at_mark(const base_implementation_type& impl,
      asio::error_code& ec) const
  {
    return socket_ops::sockatmark(impl.socket_, ec);
  }

  // Determine the number of bytes available for reading.
  std::size_t available(const base_implementation_type& impl,
      asio::error_code& ec) const
  {
    return socket_ops::available(impl.socket_, ec);
  }

  // Place the socket into the state where it will listen for new connections.
  asio::error_code listen(base_implementation_type& impl,
      int backlog, asio::error_code& ec)
  {
    socket_ops::listen(impl.socket_, backlog, ec);
    return ec;
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command>
  asio::error_code io_control(base_implementation_type& impl,
      IO_Control_Command& command, asio::error_code& ec)
  {
    socket_ops::ioctl(impl.socket_, impl.state_, command.name(),
        static_cast<ioctl_arg_type*>(command.data()), ec);
    return ec;
  }

  /// Disable sends or receives on the socket.
  asio::error_code shutdown(base_implementation_type& impl,
      socket_base::shutdown_type what, asio::error_code& ec)
  {
    socket_ops::shutdown(impl.socket_, what, ec);
    return ec;
  }

  // Send the given data to the peer.
  template <typename ConstBufferSequence>
  size_t send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    return socket_ops::sync_send(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
  }

  // Wait until data can be sent without blocking.
  size_t send(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    // Wait for socket to become ready.
    socket_ops::poll_write(impl.socket_, ec);

    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef socket_send_op<ConstBufferSequence, Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl.socket_, buffers, flags, handler);

    start_op(impl, reactor::write_op, p.p, true,
        ((impl.state_ & socket_ops::stream_oriented)
          && buffer_sequence_adapter<asio::const_buffer,
            ConstBufferSequence>::all_empty(buffers)));
    p.v = p.p = 0;
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef null_buffers_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    start_op(impl, reactor::write_op, p.p, false, false);
    p.v = p.p = 0;
  }

  // Receive some data from the peer. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  size_t receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    return socket_ops::sync_recv(impl.socket_, impl.state_,
        bufs.buffers(), bufs.count(), flags, bufs.all_empty(), ec);
  }

  // Wait until data can be received without blocking.
  size_t receive(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    // Wait for socket to become ready.
    socket_ops::poll_read(impl.socket_, ec);

    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef socket_recv_op<MutableBufferSequence, Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl.socket_, impl.state_, buffers, flags, handler);

    start_op(impl,
        (flags & socket_base::message_out_of_band)
          ? reactor::except_op : reactor::read_op,
        p.p, (flags & socket_base::message_out_of_band) == 0,
        ((impl.state_ & socket_ops::stream_oriented)
          && buffer_sequence_adapter<asio::mutable_buffer,
            MutableBufferSequence>::all_empty(buffers)));
    p.v = p.p = 0;
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive(base_implementation_type& impl, const null_buffers&,
      socket_base::message_flags flags, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef null_buffers_op<Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    start_op(impl,
        (flags & socket_base::message_out_of_band)
          ? reactor::except_op : reactor::read_op,
        p.p, false, false);
    p.v = p.p = 0;
  }

protected:
  // Start the asynchronous read or write operation.
  void start_op(base_implementation_type& impl, int op_type,
      reactor_op* op, bool non_blocking, bool noop)
  {
    if (!noop)
    {
      if ((impl.state_ & socket_ops::non_blocking)
          || socket_ops::set_internal_non_blocking(
            impl.socket_, impl.state_, op->ec_))
      {
        reactor_.start_op(op_type, impl.socket_,
            impl.reactor_data_, op, non_blocking);
        return;
      }
    }

    reactor_.post_immediate_completion(op);
  }

  // Start the asynchronous accept operation.
  void start_accept_op(base_implementation_type& impl,
      reactor_op* op, bool peer_is_open)
  {
    if (!peer_is_open)
      start_op(impl, reactor::read_op, op, true, false);
    else
    {
      op->ec_ = asio::error::already_open;
      reactor_.post_immediate_completion(op);
    }
  }

  // The selector that performs event demultiplexing for the service.
  reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_BASE_HPP
