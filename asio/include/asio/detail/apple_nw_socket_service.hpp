//
// detail/apple_nw_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_APPLE_NW_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_APPLE_NW_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/apple_nw_socket_connect_op.hpp"
#include "asio/detail/apple_nw_socket_service_base.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class apple_nw_socket_service :
  public execution_context_service_base<apple_nw_socket_service<Protocol> >,
  public apple_nw_socket_service_base
{
public:
  // The protocol type.
  typedef Protocol protocol_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  // The native type of a socket.
  typedef apple_nw_socket_service_base::native_handle_type native_handle_type;

  // The implementation type of the socket.
  struct implementation_type :
    apple_nw_socket_service_base::base_implementation_type
  {
    // Default constructor.
    implementation_type()
      : protocol_(endpoint_type().protocol())
    {
    }

    // The protocol associated with the socket.
    protocol_type protocol_;
  };

  // Constructor.
  apple_nw_socket_service(execution_context& context)
    : execution_context_service_base<
        apple_nw_socket_service<Protocol> >(context),
      apple_nw_socket_service_base(context)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    this->base_shutdown();
  }

  // Move-construct a new socket implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl) ASIO_NOEXCEPT
  {
    this->base_move_construct(impl, other_impl);

    impl.protocol_ = other_impl.protocol_;
    other_impl.protocol_ = endpoint_type().protocol();
  }

  // Move-assign from another socket implementation.
  void move_assign(implementation_type& impl,
      apple_nw_socket_service_base& other_service,
      implementation_type& other_impl)
  {
    this->base_move_assign(impl, other_service, other_impl);

    impl.protocol_ = other_impl.protocol_;
    other_impl.protocol_ = endpoint_type().protocol();
  }

  // Move-construct a new socket implementation from another protocol type.
  template <typename Protocol1>
  void converting_move_construct(implementation_type& impl,
      apple_nw_socket_service<Protocol1>&,
      typename apple_nw_socket_service<
        Protocol1>::implementation_type& other_impl)
  {
    this->base_move_construct(impl, other_impl);

    impl.protocol_ = protocol_type(other_impl.protocol_);
    other_impl.protocol_ = typename Protocol1::endpoint().protocol();
  }

  // Open a new socket implementation.
  asio::error_code open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    if (!do_open(impl, protocol.apple_nw_create_parameters(),
          protocol.apple_nw_max_receive_size(), ec))
      impl.protocol_ = protocol;
    return ec;
  }

  // Assign a native socket to a socket implementation.
  asio::error_code assign(implementation_type& impl,
      const protocol_type& protocol, const native_handle_type& native_socket,
      asio::error_code& ec)
  {
    if (!do_assign(impl, native_socket,
          protocol.apple_nw_max_receive_size(), ec))
      impl.protocol_ = protocol;
    return ec;
  }

  // Bind the socket to the specified local endpoint.
  asio::error_code bind(implementation_type& impl,
      const endpoint_type& endpoint, asio::error_code& ec)
  {
    return do_bind(impl, endpoint.apple_nw_create_endpoint(), ec);
  }

  // Get the local endpoint.
  endpoint_type local_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    endpoint_type endpoint;
    endpoint.apple_nw_set_endpoint(do_get_local_endpoint(impl, ec));
    endpoint.apple_nw_set_protocol(impl.protocol_);
    return endpoint;
  }

  // Get the remote endpoint.
  endpoint_type remote_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    endpoint_type endpoint;
    endpoint.apple_nw_set_endpoint(do_get_remote_endpoint(impl, ec));
    endpoint.apple_nw_set_protocol(impl.protocol_);
    return endpoint;
  }

  // Set a socket option.
  template <typename Option>
  asio::error_code set_option(implementation_type& impl,
      const Option& option, asio::error_code& ec)
  {
    return do_set_option(impl, &option, &Option::apple_nw_set, ec);
  }

  // Get a socket option.
  template <typename Option>
  asio::error_code get_option(const implementation_type& impl,
      Option& option, asio::error_code& ec) const
  {
    return do_get_option(impl, &option, &Option::apple_nw_get, ec);
  }

  // Connect the socket to the specified endpoint.
  asio::error_code connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, asio::error_code& ec)
  {
    return do_connect(impl, peer_endpoint.apple_nw_create_endpoint(), ec);
  }

  // Start an asynchronous connect.
  template <typename Handler, typename IoExecutor>
  void async_connect(implementation_type& impl,
      const endpoint_type& peer_endpoint,
      Handler& handler, const IoExecutor& io_ex)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef apple_nw_socket_connect_op<Handler, IoExecutor> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(handler, io_ex);

    ASIO_HANDLER_CREATION((scheduler_.context(), *p.p, "socket",
          &impl, reinterpret_cast<std::uintmax_t>(impl.parameters_.get()),
          "async_connect"));

    start_connect_op(impl, peer_endpoint.apple_nw_create_endpoint(),
        p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Disable sends or receives on the socket.
  asio::error_code shutdown(base_implementation_type& impl,
      socket_base::shutdown_type what, asio::error_code& ec)
  {
    return do_shutdown(impl, what, ec);
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename ConstBufferSequence>
  size_t send_to(implementation_type& /*impl*/, const ConstBufferSequence& /*buffers*/,
      const endpoint_type& /*destination*/, socket_base::message_flags /*flags*/,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be sent without blocking.
  size_t send_to(implementation_type&, const null_buffers&,
      const endpoint_type&, socket_base::message_flags,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
  void async_send_to(implementation_type& /*impl*/,
      const ConstBufferSequence& /*buffers*/,
      const endpoint_type& /*destination*/, socket_base::message_flags /*flags*/,
      Handler& handler, const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    asio::post(io_ex,
        detail::bind_handler(ASIO_MOVE_CAST(Handler)(handler),
          ec, bytes_transferred));
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler, typename IoExecutor>
  void async_send_to(implementation_type&, const null_buffers&,
      const endpoint_type&, socket_base::message_flags,
      Handler& handler, const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    asio::post(io_ex,
        detail::bind_handler(ASIO_MOVE_CAST(Handler)(handler),
          ec, bytes_transferred));
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename MutableBufferSequence>
  size_t receive_from(implementation_type& /*impl*/,
      const MutableBufferSequence& /*buffers*/,
      endpoint_type& /*sender_endpoint*/, socket_base::message_flags /*flags*/,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Wait until data can be received without blocking.
  size_t receive_from(implementation_type&, const null_buffers&,
      endpoint_type&, socket_base::message_flags,
      asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename MutableBufferSequence,
      typename Handler, typename IoExecutor>
  void async_receive_from(implementation_type& /*impl*/,
      const MutableBufferSequence& /*buffers*/, endpoint_type& /*sender_endpoint*/,
      socket_base::message_flags /*flags*/, Handler& handler,
      const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    asio::post(io_ex,
        detail::bind_handler(ASIO_MOVE_CAST(Handler)(handler),
          ec, bytes_transferred));
  }

  // Wait until data can be received without blocking.
  template <typename Handler, typename IoExecutor>
  void async_receive_from(implementation_type&, const null_buffers&,
      endpoint_type&, socket_base::message_flags,
      Handler& handler, const IoExecutor& io_ex)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    asio::post(io_ex,
        detail::bind_handler(ASIO_MOVE_CAST(Handler)(handler),
          ec, bytes_transferred));
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_APPLE_NW_SOCKET_SERVICE_HPP
