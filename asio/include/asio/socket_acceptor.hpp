//
// socket_acceptor.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_ACCEPTOR_HPP
#define ASIO_SOCKET_ACCEPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_socket_acceptor.hpp"
#include "asio/socket_acceptor_service.hpp"

namespace asio {

/// Typedef for the typical usage of socket_acceptor.
typedef basic_socket_acceptor<socket_acceptor_service<> > socket_acceptor;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_ACCEPTOR_HPP
