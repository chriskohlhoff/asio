//
// detail/channel_put_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CHANNEL_PUT_OP_HPP
#define ASIO_DETAIL_CHANNEL_PUT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/channel_op.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T, typename Handler>
class channel_put_op : public channel_op<T>
{
public:
  ASIO_DEFINE_HANDLER_PTR(channel_put_op);

  explicit channel_put_op(Handler& h)
    : channel_op<T>(&channel_put_op::do_complete),
      handler_(ASIO_MOVE_CAST(Handler)(h))
  {
  }

  template <typename T0>
  channel_put_op(ASIO_MOVE_ARG(T0) value, Handler& h)
    : channel_op<T>(ASIO_MOVE_CAST(T0)(value),
        &channel_put_op::do_complete),
      handler_(ASIO_MOVE_CAST(Handler)(h))
  {
  }

  static void do_complete(io_service_impl* owner, operation* base,
      const asio::error_code& ec,
      std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the handler object.
    channel_put_op* h(static_cast<channel_put_op*>(base));
    ptr p = { asio::detail::addressof(h->handler_), h, h };

    ASIO_HANDLER_COMPLETION((h));

    // Make a copy of the handler so that the memory can be deallocated before
    // the upcall is made. Even if we're not about to make an upcall, a
    // sub-object of the handler may be the true owner of the memory associated
    // with the handler. Consequently, a local copy of the handler is required
    // to ensure that any owning sub-object remains valid until after we have
    // deallocated the memory here.
    detail::binder1<Handler, asio::error_code>
      handler(h->handler_, ec);
    if (h->result_ == channel_op_base::operation_aborted)
      handler.arg1_ = asio::error::operation_aborted;
    else if (h->result_ == channel_op_base::broken_pipe)
      handler.arg1_ = asio::error::broken_pipe;
    p.h = asio::detail::addressof(handler.handler_);
    p.reset();

    // Make the upcall if required.
    if (owner)
    {
      fenced_block b(fenced_block::half);
      ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_));
      asio_handler_invoke_helpers::invoke(handler, handler.handler_);
      ASIO_HANDLER_INVOCATION_END;
    }
  }

private:
  Handler handler_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_CHANNEL_PUT_OP_HPP
