//
// ssl/detail/transport_op.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_TRANSPORT_OP_HPP
#define ASIO_SSL_DETAIL_TRANSPORT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_ENABLE_OLD_SSL)
# include "asio/deadline_timer.hpp"
# include "asio/detail/handler_alloc_helpers.hpp"
# include "asio/detail/handler_invoke_helpers.hpp"
# include "asio/ssl/detail/buffer_space.hpp"
# include "asio/ssl/detail/engine.hpp"
# include "asio/ssl/detail/transport.hpp"
# include "asio/write.hpp"
#endif // !defined(ASIO_ENABLE_OLD_SSL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

#if !defined(ASIO_ENABLE_OLD_SSL)

template <typename Stream, typename Handler>
class transport_op
{
public:
  transport_op(Stream& next_layer, asio::deadline_timer& pending_read,
      asio::deadline_timer& pending_write, int result,
      detail::buffer_space& space, const asio::error_code& ec,
      Handler& handler)
    : next_layer_(next_layer),
      pending_read_(pending_read),
      pending_write_(pending_write),
      result_(result),
      space_(space),
      ec_(ec),
      handler_(ASIO_MOVE_CAST(Handler)(handler))
  {
  }

  void operator()(const asio::error_code& ec,
      std::size_t bytes_transferred = 0, int start = 0)
  {
    switch (start)
    {
    case -1: // Called from initiating function.
    case 1: // Called after at least one async operation.
      switch (result_)
      {
      case detail::buffer_space::want_input:
        // The engine wants more data to be read from input. However, we cannot
        // allow more than one read operation at a time on the underlying
        // transport. The pending_read_ timer's expiry is to pos_infin if a
        // read is in progress, and neg_infin otherwise.
        if (pending_read_.expires_at() == boost::posix_time::neg_infin)
        {
          // Start reading some data from the underlying transport.
          next_layer_.async_read_some(
              asio::buffer(space_.input_buffer), *this);

          // Prevent other read operations from being started.
          pending_read_.expires_at(boost::posix_time::pos_infin);
        }
        else
        {
          // Wait until the current read operation completes.
          pending_read_.async_wait(*this);
        }
        break;

      default:
        // The SSL operation is done, but there might be some data to be
        // written to the output. If there isn't anything to write then we can
        // invoke the handler, but we have to keep in mind that this function
        // might be being called from the async operation's initiating
        // function. In this case we're not allowed to call the handler
        // directly. Instead, issue a zero-sized read so that the handler runs
        // "as-if" posted using io_service::post().
        if (ec_ || asio::buffer_size(space_.output) == 0)
        {
          if (start == -1)
          {
            next_layer_.async_read_some(
                asio::buffer(space_.input_buffer, 0), *this);
          }
          else
          {
            // Indicate that we should continue on to run handler directly.
            start = 0;
          }
          break;
        }
        // Fall through to process the pending output.

      case detail::buffer_space::want_output:
        // The engine wants some data to be written to the output. However, we
        // cannot allow more than one write operation at a time on the
        // underlying transport. The pending_write_ timer's expiry is to
        // pos_infin if a write is in progress, and neg_infin otherwise.
        if (pending_write_.expires_at() == boost::posix_time::neg_infin)
        {
          // Start writing all the data to the underlying transport.
          asio::async_write(next_layer_,
              asio::buffer(space_.output), *this);

          // Prevent other write operations from being started.
          pending_write_.expires_at(boost::posix_time::pos_infin);
        }
        else if (result_ == detail::buffer_space::want_output)
        {
          // Wait until the current write operation completes.
          pending_write_.async_wait(*this);
        }
        break;
      }

      // Yield control if an async operation was started.
      if (start) return; default:
      if (!ec_) ec_ = ec;

      switch (result_)
      {
      case detail::buffer_space::want_input:
        // Add received data to the engine's pending input.
        space_.input = asio::buffer(space_.input_buffer, bytes_transferred);

        // Release any waiting read operations.
        pending_read_.expires_at(boost::posix_time::neg_infin);
        break;

      default:
        if (ec || bytes_transferred == 0)
          break;
        // Fall through to remove the pending output.

      case detail::buffer_space::want_output:
        // Remove written data from the engine's pending output.
        space_.output = space_.output + bytes_transferred;

        // Release any waiting write operations.
        pending_write_.expires_at(boost::posix_time::neg_infin);
        break;
      }

      handler_(static_cast<const asio::error_code&>(ec_),
          static_cast<const int>(ec ? 0 : result_));
    }
  }

//private:
  Stream& next_layer_;
  asio::deadline_timer& pending_read_;
  asio::deadline_timer& pending_write_;
  int result_;
  detail::buffer_space& space_;
  asio::error_code ec_;
  Handler handler_;
};

template <typename Stream, typename Handler>
inline void* asio_handler_allocate(std::size_t size,
    transport_op<Stream, Handler>* this_handler)
{
  return asio_handler_alloc_helpers::allocate(
      size, this_handler->handler_);
}

template <typename Stream, typename Handler>
inline void asio_handler_deallocate(void* pointer, std::size_t size,
    transport_op<Stream, Handler>* this_handler)
{
  asio_handler_alloc_helpers::deallocate(
      pointer, size, this_handler->handler_);
}

template <typename Function, typename Stream, typename Handler>
inline void asio_handler_invoke(const Function& function,
    transport_op<Stream, Handler>* this_handler)
{
  asio_handler_invoke_helpers::invoke(
      function, this_handler->handler_);
}

#endif // !defined(ASIO_ENABLE_OLD_SSL)

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_TRANSPORT_OP_HPP
