//
// asio.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HPP
#define ASIO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/arg.hpp"
#include "asio/basic_demuxer.hpp"
#include "asio/basic_dgram_socket.hpp"
#include "asio/basic_locking_dispatcher.hpp"
#include "asio/basic_socket_acceptor.hpp"
#include "asio/basic_socket_connector.hpp"
#include "asio/basic_stream_socket.hpp"
#include "asio/basic_timer.hpp"
#include "asio/buffered_recv_stream.hpp"
#include "asio/buffered_send_stream.hpp"
#include "asio/buffered_stream.hpp"
#include "asio/default_error_handler.hpp"
#include "asio/demuxer.hpp"
#include "asio/dgram_socket.hpp"
#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/fixed_buffer.hpp"
#include "asio/ipv4/address.hpp"
#include "asio/ipv4/basic_host_resolver.hpp"
#include "asio/ipv4/host.hpp"
#include "asio/ipv4/host_resolver.hpp"
#include "asio/ipv4/multicast.hpp"
#include "asio/ipv4/tcp.hpp"
#include "asio/ipv4/udp.hpp"
#include "asio/is_recv_buffered.hpp"
#include "asio/is_send_buffered.hpp"
#include "asio/locking_dispatcher.hpp"
#include "asio/recv.hpp"
#include "asio/send.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_acceptor.hpp"
#include "asio/socket_base.hpp"
#include "asio/socket_connector.hpp"
#include "asio/socket_option.hpp"
#include "asio/stream_socket.hpp"
#include "asio/thread.hpp"
#include "asio/time.hpp"
#include "asio/timer.hpp"

#endif // ASIO_HPP
