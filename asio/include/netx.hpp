//
// netx.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef NETX_HPP
#define NETX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/awaitable.hpp"
#include "asio/co_spawn.hpp"
#include "asio/detached.hpp"
#include "asio/experimental/as_single.hpp"
#include "asio/generic/datagram_protocol.hpp"
#include "asio/generic/stream_protocol.hpp"
#include "asio/generic/host.hpp"
#include "asio/ip/tls_tcp.hpp"
#include "asio/use_awaitable.hpp"

namespace netx
{
  using asio::awaitable;
  using asio::co_spawn;
  using asio::detached;
  using asio::experimental::as_single_t;
  using asio::generic::host;
  using asio::use_awaitable_t;
  using generic_datagram_socket = asio::generic::datagram_protocol::socket;
  using generic_stream_socket = asio::generic::stream_protocol::socket;
  namespace ip { using asio::ip::tls_tcp; }
}

#endif // NETX_HPP
