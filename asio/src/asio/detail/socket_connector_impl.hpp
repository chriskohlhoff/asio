//
// socket_connector_impl.hpp
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

#ifndef ASIO_DETAIL_SOCKET_CONNECTOR_IMPL_HPP
#define ASIO_DETAIL_SOCKET_CONNECTOR_IMPL_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <set>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

class socket_connector_impl
  : private boost::noncopyable
{
public:
  typedef std::set<socket_type> socket_set;

  // Constructor.
  socket_connector_impl();

  // Destructor.
  ~socket_connector_impl();

  // Add a socket to the set.
  void add_socket(socket_type s);

  // Remove a socket from the set.
  void remove_socket(socket_type s);

  // Get a copy of all sockets in the set.
  void get_sockets(socket_set& sockets) const;

private:
  // Mutex to protect access to the internal data.
  mutable boost::mutex mutex_;

  // The sockets currently contained in the set.
  socket_set sockets_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SOCKET_CONNECTOR_IMPL_HPP
