//
// socket_connector_impl.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#include "asio/detail/socket_connector_impl.hpp"

#include "asio/detail/push_options.hpp"
#include <cassert>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

socket_connector_impl::
socket_connector_impl()
  : sockets_()
{
}

socket_connector_impl::
~socket_connector_impl()
{
  assert(sockets_.empty());
}

void
socket_connector_impl::
add_socket(
    socket_type s)
{
  boost::mutex::scoped_lock lock(mutex_);

  sockets_.insert(s);
}

void
socket_connector_impl::
remove_socket(
    socket_type s)
{
  boost::mutex::scoped_lock lock(mutex_);

  sockets_.erase(s);
}

void
socket_connector_impl::
get_sockets(
    socket_set& sockets) const
{
  boost::mutex::scoped_lock lock(mutex_);

  sockets = sockets_;
}

} // namespace detail
} // namespace asio
