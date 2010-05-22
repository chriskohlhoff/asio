//
// detail/socket_accept_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_ACCEPT_OP_HPP
#define ASIO_DETAIL_SOCKET_ACCEPT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/utility/addressof.hpp>
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Socket, typename Protocol>
class socket_accept_op_base : public reactor_op
{
public:
  socket_accept_op_base(socket_type socket, Socket& peer,
      const Protocol& protocol, typename Protocol::endpoint* peer_endpoint,
      bool enable_connection_aborted, func_type complete_func)
    : reactor_op(&socket_accept_op_base::do_perform, complete_func),
      socket_(socket),
      peer_(peer),
      protocol_(protocol),
      peer_endpoint_(peer_endpoint),
      enable_connection_aborted_(enable_connection_aborted)
  {
  }

  static bool do_perform(reactor_op* base)
  {
    socket_accept_op_base* o(static_cast<socket_accept_op_base*>(base));

    for (;;)
    {
      // Accept the waiting connection.
      asio::error_code ec;
      socket_holder new_socket;
      std::size_t addr_len = 0;
      std::size_t* addr_len_p = 0;
      socket_addr_type* addr = 0;
      if (o->peer_endpoint_)
      {
        addr_len = o->peer_endpoint_->capacity();
        addr_len_p = &addr_len;
        addr = o->peer_endpoint_->data();
      }
      new_socket.reset(socket_ops::accept(o->socket_, addr, addr_len_p, ec));

      // Retry operation if interrupted by signal.
      if (ec == asio::error::interrupted)
        continue;

      // Check if we need to run the operation again.
      if (ec == asio::error::would_block
          || ec == asio::error::try_again)
        return false;
      if (ec == asio::error::connection_aborted
          && !o->enable_connection_aborted_)
        return false;
#if defined(EPROTO)
      if (ec.value() == EPROTO && !o->enable_connection_aborted_)
        return false;
#endif // defined(EPROTO)

      // Transfer ownership of the new socket to the peer object.
      if (!ec)
      {
        if (o->peer_endpoint_)
          o->peer_endpoint_->resize(addr_len);
        o->peer_.assign(o->protocol_, new_socket.get(), ec);
        if (!ec)
          new_socket.release();
      }

      o->ec_ = ec;
      return true;
    }
  }

private:
  socket_type socket_;
  Socket& peer_;
  Protocol protocol_;
  typename Protocol::endpoint* peer_endpoint_;
  bool enable_connection_aborted_;
};

template <typename Socket, typename Protocol, typename Handler>
class socket_accept_op : public socket_accept_op_base<Socket, Protocol>
{
public:
  ASIO_DEFINE_HANDLER_PTR(socket_accept_op);

  socket_accept_op(socket_type socket, Socket& peer, const Protocol& protocol,
      typename Protocol::endpoint* peer_endpoint,
      bool enable_connection_aborted, Handler handler)
    : socket_accept_op_base<Socket, Protocol>(socket, peer,
        protocol, peer_endpoint, enable_connection_aborted,
        &socket_accept_op::do_complete),
      handler_(handler)
  {
  }

  static void do_complete(io_service_impl* owner, operation* base,
      asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the handler object.
    socket_accept_op* o(static_cast<socket_accept_op*>(base));
    ptr p = { boost::addressof(o->handler_), o, o };

    // Make a copy of the handler so that the memory can be deallocated before
    // the upcall is made. Even if we're not about to make an upcall, a
    // sub-object of the handler may be the true owner of the memory associated
    // with the handler. Consequently, a local copy of the handler is required
    // to ensure that any owning sub-object remains valid until after we have
    // deallocated the memory here.
    detail::binder1<Handler, asio::error_code>
      handler(o->handler_, o->ec_);
    p.h = boost::addressof(handler.handler_);
    p.reset();

    // Make the upcall if required.
    if (owner)
    {
      asio::detail::fenced_block b;
      asio_handler_invoke_helpers::invoke(handler, handler.handler_);
    }
  }

private:
  Handler handler_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_ACCEPT_OP_HPP
