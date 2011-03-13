//
// ssl/detail/transport.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_TRANSPORT_HPP
#define ASIO_SSL_DETAIL_TRANSPORT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(ASIO_ENABLE_OLD_SSL)
# include "asio/deadline_timer.hpp"
# include "asio/ssl/detail/buffer_space.hpp"
# include "asio/ssl/detail/transport_op.hpp"
# include "asio/write.hpp"
# include <boost/type_traits/remove_reference.hpp>
#endif // !defined(ASIO_ENABLE_OLD_SSL)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

#if !defined(ASIO_ENABLE_OLD_SSL)

template <typename Stream>
class transport
{
public:
  // The type of the next layer.
  typedef typename boost::remove_reference<Stream>::type next_layer_type;

  // The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  // Constructor initialises the underlying stream.
  template <typename Arg>
  explicit transport(Arg& arg)
    : next_layer_(arg),
      pending_read_(next_layer_.lowest_layer().get_io_service()),
      pending_write_(next_layer_.lowest_layer().get_io_service())
  {
    pending_read_.expires_at(boost::posix_time::neg_infin);
    pending_write_.expires_at(boost::posix_time::neg_infin);
  }

  asio::io_service& get_io_service()
  {
    return next_layer_.lowest_layer().get_io_service();
  }

  next_layer_type& next_layer()
  {
    return next_layer_;
  }

  const next_layer_type& next_layer() const
  {
    return next_layer_;
  }

  lowest_layer_type& lowest_layer()
  {
    return next_layer_.lowest_layer();
  }

  const lowest_layer_type& lowest_layer() const
  {
    return next_layer_.lowest_layer();
  }

  int sync(int result, buffer_space& space, asio::error_code& ec)
  {
    switch (result)
    {
    case detail::buffer_space::want_input:
      space.input = asio::buffer(space.input_buffer,
          next_layer_.read_some(asio::buffer(space.input_buffer), ec));
      break;
    default:
      if (ec || asio::buffer_size(space.output) == 0)
        break;
    case detail::buffer_space::want_output:
      space.output =
        space.output + asio::write(next_layer_,
          asio::buffer(space.output), ec);
      break;
    }
    return ec ? 0 : result;
  }

  template <typename Handler>
  void async(int result, buffer_space& space,
      const asio::error_code& ec, int start, Handler& handler)
  {
    transport_op<next_layer_type, Handler>(
      next_layer_, pending_read_, pending_write_, result, space, ec, handler)(
        asio::error_code(), 0, start ? -1 : 1);
  }

private:
  Stream next_layer_;
  asio::deadline_timer pending_read_;
  asio::deadline_timer pending_write_;
};

#endif // !defined(ASIO_ENABLE_OLD_SSL)

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_TRANSPORT_HPP
