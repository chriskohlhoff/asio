//
// socket_connector.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SOCKET_CONNECTOR_HPP
#define ASIO_SOCKET_CONNECTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/basic_socket_connector.hpp"
#include "asio/socket_connector_service.hpp"

namespace asio {

/// Typedef for the typical usage of socket_connector.
typedef basic_socket_connector<socket_connector_service<> > socket_connector;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SOCKET_CONNECTOR_HPP
