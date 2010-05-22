//
// detail/socket_connect_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SOCKET_CONNECT_OP_HPP
#define ASIO_DETAIL_SOCKET_CONNECT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/utility/addressof.hpp>
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/reactor_op.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class socket_connect_op_base : public reactor_op
{
public:
  socket_connect_op_base(socket_type socket, func_type complete_func)
    : reactor_op(&socket_connect_op_base::do_perform, complete_func),
      socket_(socket)
  {
  }

  static bool do_perform(reactor_op* base)
  {
    socket_connect_op_base* o(static_cast<socket_connect_op_base*>(base));
    return socket_ops::non_blocking_connect(o->socket_, o->ec_);
  }

private:
  socket_type socket_;
};

template <typename Handler>
class socket_connect_op : public socket_connect_op_base
{
public:
  ASIO_DEFINE_HANDLER_PTR(socket_connect_op);

  socket_connect_op(socket_type socket, Handler handler)
    : socket_connect_op_base(socket, &socket_connect_op::do_complete),
      handler_(handler)
  {
  }

  static void do_complete(io_service_impl* owner, operation* base,
      asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the handler object.
    socket_connect_op* o(static_cast<socket_connect_op*>(base));
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
      asio_handler_invoke_helpers::invoke(handler, handler);
    }
  }

private:
  Handler handler_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_CONNECT_OP_HPP
