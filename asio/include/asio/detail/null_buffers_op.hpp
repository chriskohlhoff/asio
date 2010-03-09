//
// null_buffers_op.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NULL_BUFFERS_OP_HPP
#define ASIO_DETAIL_NULL_BUFFERS_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/reactor_op.hpp"

namespace asio {
namespace detail {

template <typename Handler>
class null_buffers_op : public reactor_op
{
public:
  null_buffers_op(Handler handler)
    : reactor_op(&null_buffers_op::do_perform, &null_buffers_op::do_complete),
      handler_(handler)
  {
  }

  static bool do_perform(reactor_op*)
  {
    return true;
  }

  static void do_complete(io_service_impl* owner, operation* base,
      asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
  {
    // Take ownership of the handler object.
    null_buffers_op* o(static_cast<null_buffers_op*>(base));
    typedef handler_alloc_traits<Handler, null_buffers_op> alloc_traits;
    handler_ptr<alloc_traits> ptr(o->handler_, o);

    // Make the upcall if required.
    if (owner)
    {
      // Make a copy of the handler so that the memory can be deallocated
      // before the upcall is made. Even if we're not about to make an upcall,
      // a sub-object of the handler may be the true owner of the memory
      // associated with the handler. Consequently, a local copy of the handler
      // is required to ensure that any owning sub-object remains valid until
      // after we have deallocated the memory here.
      detail::binder2<Handler, asio::error_code, std::size_t>
        handler(o->handler_, o->ec_, o->bytes_transferred_);
      ptr.reset();
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

#endif // ASIO_DETAIL_NULL_BUFFERS_OP_HPP
