//
// detail/apple_nw_acceptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_APPLE_NW_ACCEPTOR_SERVICE_HPP
#define ASIO_DETAIL_APPLE_NW_ACCEPTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/apple_nw_acceptor_service_base.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"
#include <Network/Network.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class apple_nw_acceptor_service :
  public execution_context_service_base<apple_nw_acceptor_service<Protocol> >,
  public apple_nw_acceptor_service_base
{
public:
  // The protocol type.
  typedef Protocol protocol_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  // The native type of a socket.
  typedef apple_nw_acceptor_service_base::native_handle_type native_handle_type;

  // The implementation type of the acceptor.
  struct implementation_type :
    apple_nw_acceptor_service_base::base_implementation_type
  {
    // Default constructor.
    implementation_type()
      : protocol_(endpoint_type().protocol())
    {
    }

    // The protocol associated with the acceptor.
    protocol_type protocol_;
  };

  // Constructor.
  apple_nw_acceptor_service(execution_context& context)
    : execution_context_service_base<
        apple_nw_acceptor_service<Protocol> >(context),
      apple_nw_acceptor_service_base(context)
  {
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown()
  {
    this->base_shutdown();
  }

  // Move-construct a new acceptor implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl) ASIO_NOEXCEPT
  {
    this->base_move_construct(impl, other_impl);

    impl.protocol_ = other_impl.protocol_;
    other_impl.protocol_ = endpoint_type().protocol();
  }

  // Move-assign from another acceptor implementation.
  void move_assign(implementation_type& impl,
      apple_nw_acceptor_service_base& other_service,
      implementation_type& other_impl)
  {
    this->base_move_assign(impl, other_service, other_impl);

    impl.protocol_ = other_impl.protocol_;
    other_impl.protocol_ = endpoint_type().protocol();
  }

  // Move-construct a new acceptor implementation from another protocol type.
  template <typename Protocol1>
  void converting_move_construct(implementation_type& impl,
      apple_nw_acceptor_service<Protocol1>&,
      typename apple_nw_acceptor_service<
        Protocol1>::implementation_type& other_impl)
  {
    this->base_move_construct(impl, other_impl);

    impl.protocol_ = protocol_type(other_impl.protocol_);
    other_impl.protocol_ = typename Protocol1::endpoint().protocol();
  }

  // Open a new acceptor implementation.
  asio::error_code open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    if (!do_open(impl, protocol.apple_nw_create_parameters(), ec))
      impl.protocol_ = protocol;
    return ec;
  }

  // Assign a native acceptor to a acceptor implementation.
  asio::error_code assign(implementation_type& impl,
      const protocol_type& protocol, const native_handle_type& native_acceptor,
      asio::error_code& ec)
  {
    if (!do_assign(impl, native_acceptor, ec))
      impl.protocol_ = protocol;
    return ec;
  }

  // Bind the acceptor to the specified local endpoint.
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

  // Set a acceptor option.
  template <typename Option>
  asio::error_code set_option(implementation_type& impl,
      const Option& option, asio::error_code& ec)
  {
    return do_set_option(impl, &option, &Option::apple_nw_set, ec);
  }

  // Get a acceptor option.
  template <typename Option>
  asio::error_code get_option(const implementation_type& impl,
      Option& option, asio::error_code& ec) const
  {
    return do_get_option(impl, &option, &Option::apple_nw_get, ec);
  }

  // Accept a new connection.
  template <typename Socket>
  asio::error_code accept(implementation_type& impl,
      Socket& peer, endpoint_type* peer_endpoint, asio::error_code& ec)
  {
    // We cannot accept a socket that is already open.
    if (peer.is_open())
    {
      ec = asio::error::already_open;
      return ec;
    }

    apple_nw_ptr<nw_connection_t> new_connection(do_accept(impl, ec));

    // On success, assign new connection to peer socket object.
    if (new_connection && !ec)
    {
      typename Socket::native_handle_type handle(new_connection.get());
      peer.assign(impl.protocol_, handle, ec);
      if (!ec)
        new_connection.release();

      if (peer_endpoint && !ec)
      {
        *peer_endpoint = peer.remote_endpoint(ec);
        if (ec)
          peer.close();
      }
    }

    return ec;
  }

  // Start an asynchronous accept. The peer and peer_endpoint objects must be
  // valid until the accept's handler is invoked.
  template <typename Socket, typename Handler, typename IoExecutor>
  void async_accept(implementation_type& impl, Socket& peer,
      endpoint_type* peer_endpoint, Handler& handler, const IoExecutor& io_ex)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef apple_nw_socket_accept_op<Protocol, Socket, Handler, IoExecutor> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(peer, impl.protocol_, peer_endpoint, handler, io_ex);

    ASIO_HANDLER_CREATION((scheduler_.context(), *p.p, "socket",
          &impl, reinterpret_cast<std::uintmax_t>(impl.parameters_.get()),
          "async_accept"));

    start_accept_op(impl, p.p, is_continuation, peer.is_open());
    p.v = p.p = 0;
  }

#if defined(ASIO_HAS_MOVE)
  // Start an asynchronous accept. The peer_endpoint object must be valid until
  // the accept's handler is invoked.
  template <typename PeerIoExecutor, typename Handler, typename IoExecutor>
  void async_move_accept(implementation_type& impl,
      const PeerIoExecutor& peer_io_ex, endpoint_type* peer_endpoint,
      Handler& handler, const IoExecutor& io_ex)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef apple_nw_socket_move_accept_op<Protocol,
        PeerIoExecutor, Handler, IoExecutor> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      op::ptr::allocate(handler), 0 };
    p.p = new (p.v) op(peer_io_ex, impl.protocol_,
        peer_endpoint, handler, io_ex);

    ASIO_HANDLER_CREATION((scheduler_.context(), *p.p, "socket",
          &impl, reinterpret_cast<std::uintmax_t>(impl.parameters_.get()),
          "async_accept"));

    start_accept_op(impl, p.p, is_continuation, false);
    p.v = p.p = 0;
  }
#endif // defined(ASIO_HAS_MOVE)
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_APPLE_NW_ACCEPTOR_SERVICE_HPP
