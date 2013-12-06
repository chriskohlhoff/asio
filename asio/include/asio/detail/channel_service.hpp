//
// detail/channel_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2013 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_CHANNEL_SERVICE_HPP
#define ASIO_DETAIL_CHANNEL_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <deque>
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/addressof.hpp"
#include "asio/detail/channel_get_op.hpp"
#include "asio/detail/channel_op.hpp"
#include "asio/detail/channel_put_op.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class channel_service
  : public asio::detail::service_base<channel_service>
{
public:
  // Possible states for a channel.
  enum state
  {
    get_block_put_block,
    get_block_put_buffer, 
    get_block_put_waiter,
    get_buffer_put_buffer,
    get_buffer_put_block,
    get_buffer_put_closed,
    get_waiter_put_block,
    get_waiter_put_closed,
    closed
  };

  // The base implementation type of all channels.
  struct base_implementation_type
  {
    // Default constructor.
    base_implementation_type()
      : state_(get_block_put_block),
        max_buffer_size_(0),
        next_(0),
        prev_(0)
    {
    }

    // The current state of the channel.
    state state_;

    // The maximum number of elements that may be buffered in the channel.
    std::size_t max_buffer_size_;

    // The operations that are waiting on the channel.
    op_queue<channel_op_base> waiters_;

    // Pointers to adjacent channel implementations in linked list.
    base_implementation_type* next_;
    base_implementation_type* prev_;
  };

  // The implementation for a specific value type.
  template <typename T>
  struct implementation_type;

  // Constructor.
  ASIO_DECL channel_service(
      asio::io_service& io_service);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown_service();

  // Construct a new channel implementation.
  ASIO_DECL void construct(base_implementation_type&,
      std::size_t max_buffer_size);

  // Destroy a channel implementation.
  ASIO_DECL void destroy(base_implementation_type& impl);

  // Determine whether the channel is open.
  bool is_open(const base_implementation_type& impl) const;

  // Open the channel.
  void open(base_implementation_type& impl);

  // Close the channel.
  ASIO_DECL void close(base_implementation_type& impl);

  // Cancel all operations associated with the channel.
  ASIO_DECL void cancel(base_implementation_type& impl);

  // Determine whether a value can be read from the channel without blocking.
  template <typename T>
  bool ready(const implementation_type<T>& impl) const;

  // Determine whether a value can be read from the channel without blocking.
  bool ready(const implementation_type<void>& impl) const;

  // Synchronously place a new value into the channel.
  template <typename T, typename T0>
  void put(implementation_type<T>& impl,
      ASIO_MOVE_ARG(T0) value, asio::error_code& ec);

  // Synchronously place a new void "value" into the channel.
  void put(implementation_type<void>& impl, asio::error_code& ec);

  // Asynchronously place a new value into the channel.
  template <typename T, typename T0, typename Handler>
  void async_put(implementation_type<T>& impl,
      ASIO_MOVE_ARG(T0) value, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef channel_put_op<T, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(ASIO_MOVE_CAST(T0)(value), handler);

    ASIO_HANDLER_CREATION((p.p, "channel", this, "async_put"));

    start_put_op(impl, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Asynchronously place a new value into the channel.
  template <typename T, typename Handler>
  void async_put(implementation_type<T>& impl, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef channel_put_op<T, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    ASIO_HANDLER_CREATION((p.p, "channel", this, "async_put"));

    start_put_op(impl, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Synchronously remove a value from the channel.
  template <typename T>
  T get(implementation_type<T>& impl, asio::error_code& ec);

  // Synchronously remove a void "value" from the channel.
  void get(implementation_type<void>& impl, asio::error_code& ec);

  // Asynchronously remove a value from the channel.
  template <typename T, typename Handler>
  void async_get(implementation_type<T>& impl, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef channel_get_op<T, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    ASIO_HANDLER_CREATION((p.p, "channel", this, "async_get"));

    start_get_op(impl, p.p, is_continuation);
    p.v = p.p = 0;
  }

private:
  // Helper function to start an asynchronous put operation.
  template <typename T>
  void start_put_op(implementation_type<T>& impl,
      channel_op<T>* putter, bool is_continuation);

  // Helper function to start an asynchronous put operation.
  void start_put_op(implementation_type<void>& impl,
      channel_op<void>* putter, bool is_continuation);

  // Helper function to start an asynchronous get operation.
  template <typename T>
  void start_get_op(implementation_type<T>& impl,
      channel_op<T>* getter, bool is_continuation);

  // Helper function to start an asynchronous get operation.
  void start_get_op(implementation_type<void>& impl,
      channel_op<void>* getter, bool is_continuation);

  // The io_service implementation used for delivering completions.
  io_service_impl& io_service_;

  // Mutex to protect access to the linked list of implementations. 
  asio::detail::mutex mutex_;

  // The head of a linked list of all implementations.
  base_implementation_type* impl_list_;
};

// The implementation for a specific value type.
template <typename T>
struct channel_service::implementation_type : base_implementation_type
{
  // Buffered values.
  std::deque<T> buffer_;
};

// The implementation for a void value type.
template <>
struct channel_service::implementation_type<void> : base_implementation_type
{
  implementation_type() : buffered_(0) {}

  // Number of buffered "values".
  std::size_t buffered_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

//#include "asio/detail/impl/channel_service.hpp"
#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/channel_service.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_CHANNEL_SERVICE_HPP
