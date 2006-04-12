//
// basic_datagram_socket.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_DATAGRAM_SOCKET_HPP
#define ASIO_BASIC_DATAGRAM_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_socket.hpp"

namespace asio {

/// Provides datagram-oriented socket functionality.
/**
 * The basic_datagram_socket class template provides asynchronous and blocking
 * datagram-oriented socket functionality.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object, Error_Source.
 */
template <typename Service>
class basic_datagram_socket
  : public basic_socket<Service>
{
public:
  /// The native representation of a socket.
  typedef typename Service::native_type native_type;

  /// The protocol type.
  typedef typename Service::protocol_type protocol_type;

  /// The endpoint type.
  typedef typename Service::endpoint_type endpoint_type;

  /// Construct a basic_datagram_socket without opening it.
  explicit basic_datagram_socket(asio::io_service& io_service);

  /// Construct and open a basic_datagram_socket.
  basic_datagram_socket(asio::io_service& io_service,
      const protocol_type& protocol);

  /// Construct a basic_datagram_socket, opening it and binding it to the given
  /// local endpoint.
  basic_datagram_socket(asio::io_service& io_service,
      const endpoint_type& endpoint);

  /// Construct a basic_datagram_socket on an existing native socket.
  basic_datagram_socket(asio::io_service& io_service,
      const native_type& native_socket);

  /// Send some data on a connected socket.
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers);

  /// Send some data on a connected socket.
  template <typename Const_Buffers>
  std::size_t send(const Const_Buffers& buffers,
      socket_base::message_flags flags);

  /// Send some data on a connected socket.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send(const Const_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler);

  /// Start an asynchronous send on a connected socket.
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers, Handler handler);

  /// Start an asynchronous send on a connected socket.
  template <typename Const_Buffers, typename Handler>
  void async_send(const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler);

  /// Send a datagram to the specified endpoint.
  template <typename Const_Buffers>
  std::size_t send_to(const Const_Buffers& buffers,
      const endpoint_type& destination);

  /// Send a datagram to the specified endpoint.
  template <typename Const_Buffers>
  std::size_t send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, socket_base::message_flags flags);

  /// Send a datagram to the specified endpoint.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, socket_base::message_flags flags,
      Error_Handler error_handler);

  /// Start an asynchronous send.
  template <typename Const_Buffers, typename Handler>
  void async_send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, Handler handler);

  /// Start an asynchronous send.
  template <typename Const_Buffers, typename Handler>
  void async_send_to(const Const_Buffers& buffers,
      const endpoint_type& destination, socket_base::message_flags flags,
      Handler handler);

  /// Receive some data on a connected socket.
  template <typename Mutable_Buffers>
  std::size_t receive(const Mutable_Buffers& buffers);

  /// Receive some data on a connected socket.
  template <typename Mutable_Buffers>
  std::size_t receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags);

  /// Receive some data on a connected socket.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler);

  /// Start an asynchronous receive on a connected socket.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers, Handler handler);

  /// Start an asynchronous receive on a connected socket.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler);

  /// Receive a datagram with the endpoint of the sender.
  template <typename Mutable_Buffers>
  std::size_t receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint);
  
  /// Receive a datagram with the endpoint of the sender.
  template <typename Mutable_Buffers>
  std::size_t receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags);
  
  /// Receive a datagram with the endpoint of the sender.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags,
      Error_Handler error_handler);

  /// Start an asynchronous receive.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, Handler handler);

  /// Start an asynchronous receive.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive_from(const Mutable_Buffers& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags,
      Handler handler);
};

} // namespace asio

#include "asio/impl/basic_datagram_socket.ipp"

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DATAGRAM_SOCKET_HPP
