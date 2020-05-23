//
// detail/apple_nw_socket_service_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_APPLE_NW_SOCKET_SERVICE_BASE_HPP
#define ASIO_DETAIL_APPLE_NW_SOCKET_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/post.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/apple_nw_async_scope.hpp"
#include "asio/detail/apple_nw_async_op.hpp"
#include "asio/detail/apple_nw_buffer_helpers.hpp"
#include "asio/detail/apple_nw_ptr.hpp"
#include "asio/detail/apple_nw_socket_send_op.hpp"
#include "asio/detail/apple_nw_socket_recv_op.hpp"
#include "asio/detail/atomic_count.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/scheduler.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/detail/socket_types.hpp"
#include <Network/Network.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class apple_nw_socket_service_base
{
public:
  // The native type of a socket.
  struct native_handle_type
  {
    explicit native_handle_type(nw_connection_t c)
      : parameters(0),
        connection(c)
    {
    }

    native_handle_type(nw_parameters_t p, nw_connection_t c)
      : parameters(p),
        connection(c)
    {
    }

    nw_parameters_t parameters;
    nw_connection_t connection;
  };

  // The implementation type of the socket.
  struct base_implementation_type
  {
    // Default constructor.
    base_implementation_type()
      : parameters_(),
        connection_(),
        next_(0),
        prev_(0)
    {
    }

    // The parameters to be used to create the connection.
    apple_nw_ptr<nw_parameters_t> parameters_;

    // The underlying native connection.
    apple_nw_ptr<nw_connection_t> connection_;

    // Override for maximum message size. Set to 65535 for UDP so that entire
    // packets are received, and any excess data is discarded when copying to
    // the buffer sequence. Otherwise set to 0.
    std::size_t max_receive_size_;

    // Pointers to adjacent socket implementations in linked list.
    base_implementation_type* next_;
    base_implementation_type* prev_;
  };

  // Constructor.
  ASIO_DECL apple_nw_socket_service_base(execution_context& context);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void base_shutdown();

  // Construct a new socket implementation.
  ASIO_DECL void construct(base_implementation_type&);

  // Move-construct a new socket implementation.
  ASIO_DECL void base_move_construct(base_implementation_type& impl,
      base_implementation_type& other_impl) ASIO_NOEXCEPT;

  // Move-assign from another socket implementation.
  ASIO_DECL void base_move_assign(base_implementation_type& impl,
      apple_nw_socket_service_base& other_service,
      base_implementation_type& other_impl);

  // Destroy a socket implementation.
  ASIO_DECL void destroy(base_implementation_type& impl);

  // Determine whether the socket is open.
  bool is_open(const base_implementation_type& impl) const
  {
    return !!impl.parameters_;
  }

  // Destroy a socket implementation.
  ASIO_DECL asio::error_code close(
      base_implementation_type& impl, asio::error_code& ec);

  // Release ownership of the socket.
  ASIO_DECL native_handle_type release(
      base_implementation_type& impl, asio::error_code& ec);

  // Get the native socket representation.
  native_handle_type native_handle(base_implementation_type& impl)
  {
    return native_handle_type(impl.parameters_, impl.connection_);
  }

  // Cancel all operations associated with the socket.
  ASIO_DECL asio::error_code cancel(
      base_implementation_type&, asio::error_code& ec);

  // Determine whether the socket is at the out-of-band data mark.
  bool at_mark(const base_implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return false;
  }

  // Determine the number of bytes available for reading.
  std::size_t available(const base_implementation_type&,
      asio::error_code& ec) const
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command>
  asio::error_code io_control(base_implementation_type&,
      IO_Control_Command&, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Gets the non-blocking mode of the socket.
  bool non_blocking(const base_implementation_type&) const
  {
    return false;
  }

  // Sets the non-blocking mode of the socket.
  asio::error_code non_blocking(base_implementation_type&,
      bool, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Gets the non-blocking mode of the native socket implementation.
  bool native_non_blocking(const base_implementation_type&) const
  {
    return false;
  }

  // Sets the non-blocking mode of the native socket implementation.
  asio::error_code native_non_blocking(base_implementation_type&,
      bool, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Send the given data to the peer.
  template <typename ConstBufferSequence>
  std::size_t send(base_implementation_type& impl,
      const ConstBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    return do_send(impl, apple_nw_buffers_to_dispatch_data(buffers),
        asio::buffer_size(buffers), flags, ec);
  }

  // Wait until data can be sent without blocking.
  std::size_t send(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
  void async_send(base_implementation_type& impl,
      const ConstBufferSequence& buffers, socket_base::message_flags flags,
      Handler& handler, const IoExecutor& io_ex)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef apple_nw_socket_send_op<
        ConstBufferSequence, Handler, IoExecutor> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler, io_ex);

    ASIO_HANDLER_CREATION((scheduler_.context(), *p.p, "socket",
          &impl, reinterpret_cast<std::uintmax_t>(impl.parameters_.get()),
          "async_send"));

    start_send_op(impl, apple_nw_buffers_to_dispatch_data(buffers),
        asio::buffer_size(buffers), flags, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler, typename IoExecutor>
  void async_send(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    asio::post(io_ex,
        detail::bind_handler(ASIO_MOVE_CAST(Handler)(handler),
          ec, bytes_transferred));
  }

  // Receive some data from the peer. Returns the number of bytes received.
  template <typename MutableBufferSequence>
  std::size_t receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    return apple_nw_buffers_from_dispatch_data(buffers,
        do_receive(impl, asio::buffer_size(buffers), flags, ec));
  }

  // Wait until data can be received without blocking.
  std::size_t receive(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence,
      typename Handler, typename IoExecutor>
  void async_receive(base_implementation_type& impl,
      const MutableBufferSequence& buffers, socket_base::message_flags flags,
      Handler& handler, const IoExecutor& io_ex)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef apple_nw_socket_recv_op<
        MutableBufferSequence, Handler, IoExecutor> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(buffers, handler, io_ex);

    ASIO_HANDLER_CREATION((scheduler_.context(), *p.p, "socket",
          &impl, reinterpret_cast<std::uintmax_t>(impl.parameters_.get()),
          "async_receive"));

    start_receive_op(impl, asio::buffer_size(buffers),
        flags, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Wait until data can be received without blocking.
  template <typename Handler, typename IoExecutor>
  void async_receive(base_implementation_type&, const null_buffers&,
      socket_base::message_flags, Handler& handler, const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    asio::post(io_ex,
        detail::bind_handler(ASIO_MOVE_CAST(Handler)(handler),
          ec, bytes_transferred));
  }

  // Wait for the socket to become ready to read, ready to write, or to have
  // pending error conditions.
  asio::error_code wait(base_implementation_type&,
      socket_base::wait_type, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  // Asynchronously wait for the socket to become ready to read, ready to
  // write, or to have pending error conditions.
  template <typename Handler, typename IoExecutor>
  void async_wait(base_implementation_type&,
      socket_base::wait_type, Handler& handler, const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    asio::post(io_ex, detail::bind_handler(
          ASIO_MOVE_CAST(Handler)(handler), ec));
  }

protected:
  // Open a new socket implementation.
  ASIO_DECL asio::error_code do_open(base_implementation_type& impl,
      apple_nw_ptr<nw_parameters_t> parameters,
      std::size_t max_receive_size, asio::error_code& ec);

  // Assign a native socket to a socket implementation.
  ASIO_DECL asio::error_code do_assign(base_implementation_type& impl,
      const native_handle_type& native_socket,
      std::size_t max_receive_size, asio::error_code& ec);

  // Helper function to obtain local endpoint associated with the connection.
  ASIO_DECL apple_nw_ptr<nw_endpoint_t> do_get_local_endpoint(
      const base_implementation_type& impl,
      asio::error_code& ec) const;

  // Helper function to obtain local endpoint associated with the connection.
  ASIO_DECL apple_nw_ptr<nw_endpoint_t> do_get_remote_endpoint(
      const base_implementation_type& impl,
      asio::error_code& ec) const;

  // Helper function to set a socket option.
  ASIO_DECL asio::error_code do_set_option(
      base_implementation_type& impl, const void* option,
      void (*set_fn)(const void*, nw_parameters_t,
        nw_connection_t, asio::error_code&),
      asio::error_code& ec);

  // Helper function to get a socket option.
  ASIO_DECL asio::error_code do_get_option(
      const base_implementation_type& impl, void* option,
      void (*get_fn)(void*, nw_parameters_t,
        nw_connection_t, asio::error_code&),
      asio::error_code& ec) const;

  // Helper function to perform a synchronous connect.
  ASIO_DECL asio::error_code do_bind(
      base_implementation_type& impl,
      apple_nw_ptr<nw_endpoint_t> endpoint,
      asio::error_code& ec);

  // Helper function to perform a synchronous connect.
  ASIO_DECL asio::error_code do_connect(
      base_implementation_type& impl,
      apple_nw_ptr<nw_endpoint_t> endpoint,
      asio::error_code& ec);

  // Helper function to start an asynchronous connect.
  ASIO_DECL void start_connect_op(base_implementation_type& impl,
      apple_nw_ptr<nw_endpoint_t> endpoint, apple_nw_async_op<void>* op,
      bool is_continuation);

  // Helper function to perform a shutdown.
  ASIO_DECL asio::error_code do_shutdown(
      base_implementation_type& impl,
      socket_base::shutdown_type what, asio::error_code& ec);

  // Helper function to perform a synchronous send.
  ASIO_DECL std::size_t do_send(base_implementation_type& impl,
      apple_nw_ptr<dispatch_data_t> data, std::size_t data_size,
      socket_base::message_flags flags, asio::error_code& ec);

  // Helper function to start an asynchronous send.
  ASIO_DECL void start_send_op(base_implementation_type& impl,
      apple_nw_ptr<dispatch_data_t> data, std::size_t data_size,
      socket_base::message_flags flags, apple_nw_async_op<void>* op,
      bool is_continuation);

  // Helper function to perform a synchronous receive.
  ASIO_DECL apple_nw_ptr<dispatch_data_t> do_receive(
      base_implementation_type& impl, std::size_t max_size,
      socket_base::message_flags flags, asio::error_code& ec);

  // Helper function to start an asynchronous receive.
  ASIO_DECL void start_receive_op(base_implementation_type& impl,
      std::size_t max_size, socket_base::message_flags flags,
      apple_nw_async_op<apple_nw_ptr<dispatch_data_t> >* op,
      bool is_continuation);

  // The scheduler implementation used for delivering completions.
  scheduler& scheduler_;

  // Mutex to protect access to the linked list of implementations.
  asio::detail::mutex mutex_;

  // The head of a linked list of all implementations.
  base_implementation_type* impl_list_;

  // Used to wait for outstanding operations to complete.
  apple_nw_async_scope async_scope_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/apple_nw_socket_service_base.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_APPLE_NW_SOCKET_SERVICE_BASE_HPP
