//
// detail/impl/apple_nw_acceptor_service_base.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_APPLE_NW_ACCEPTOR_SERVICE_BASE_IPP
#define ASIO_DETAIL_IMPL_APPLE_NW_ACCEPTOR_SERVICE_BASE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include <cstring>
#include <string>
#include "asio/detail/apple_nw_acceptor_service_base.hpp"
#include "asio/detail/apple_nw_sync_result.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

apple_nw_acceptor_service_base::apple_nw_acceptor_service_base(
    execution_context& context)
  : scheduler_(use_service<scheduler>(context)),
    mutex_(),
    impl_list_(0)
{
}

void apple_nw_acceptor_service_base::base_shutdown()
{
  // Close all implementations, causing all operations to complete.
  op_queue<scheduler_operation> abandoned_ops;
  asio::detail::mutex::scoped_lock lock(mutex_);
  base_implementation_type* impl = impl_list_;
  while (impl)
  {
    if (impl->accept_queues_)
    {
      std::unique_lock<std::mutex> lock(impl->accept_queues_->mutex_);
      while (!impl->accept_queues_->pending_async_.empty())
      {
        scheduler_operation* op = impl->accept_queues_->pending_async_.front();
        impl->accept_queues_->pending_async_.pop();
        abandoned_ops.push(op);
      }
      impl->accept_queues_.reset();
    }
    if (impl->listener_)
    {
      nw_listener_cancel(impl->listener_);
    }
    impl = impl->next_;
  }
  scheduler_.abandon_operations(abandoned_ops);
}

void apple_nw_acceptor_service_base::construct(
    apple_nw_acceptor_service_base::base_implementation_type& impl)
{
  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void apple_nw_acceptor_service_base::base_move_construct(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    apple_nw_acceptor_service_base::base_implementation_type& other_impl)
  ASIO_NOEXCEPT
{
  impl.parameters_.swap(other_impl.parameters_);
  other_impl.parameters_.reset();
  impl.listener_.swap(other_impl.listener_);
  other_impl.listener_.reset();
  impl.accept_queues_.swap(other_impl.accept_queues_);
  other_impl.accept_queues_.reset();

  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void apple_nw_acceptor_service_base::base_move_assign(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    apple_nw_acceptor_service_base& other_service,
    apple_nw_acceptor_service_base::base_implementation_type& other_impl)
{
  asio::error_code ignored_ec;
  close(impl, ignored_ec);

  if (this != &other_service)
  {
    // Remove implementation from linked list of all implementations.
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (impl_list_ == &impl)
      impl_list_ = impl.next_;
    if (impl.prev_)
      impl.prev_->next_ = impl.next_;
    if (impl.next_)
      impl.next_->prev_= impl.prev_;
    impl.next_ = 0;
    impl.prev_ = 0;
  }

  impl.parameters_.swap(other_impl.parameters_);
  other_impl.parameters_.reset();
  impl.listener_.swap(other_impl.listener_);
  other_impl.listener_.reset();
  impl.accept_queues_.swap(other_impl.accept_queues_);
  other_impl.accept_queues_.reset();

  if (this != &other_service)
  {
    // Insert implementation into linked list of all implementations.
    asio::detail::mutex::scoped_lock lock(other_service.mutex_);
    impl.next_ = other_service.impl_list_;
    impl.prev_ = 0;
    if (other_service.impl_list_)
      other_service.impl_list_->prev_ = &impl;
    other_service.impl_list_ = &impl;
  }
}

void apple_nw_acceptor_service_base::destroy(
    apple_nw_acceptor_service_base::base_implementation_type& impl)
{
  asio::error_code ignored_ec;
  close(impl, ignored_ec);

  // Remove implementation from linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  if (impl_list_ == &impl)
    impl_list_ = impl.next_;
  if (impl.prev_)
    impl.prev_->next_ = impl.next_;
  if (impl.next_)
    impl.next_->prev_= impl.prev_;
  impl.next_ = 0;
  impl.prev_ = 0;
}

asio::error_code apple_nw_acceptor_service_base::close(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (impl.listener_)
  {
    nw_listener_cancel(impl.listener_);
  }

  impl.parameters_.reset();
  impl.listener_.reset();
  impl.accept_queues_.reset();

  ec = asio::error_code();
  return ec;
}

apple_nw_acceptor_service_base::native_handle_type
apple_nw_acceptor_service_base::release(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    return native_handle_type(0, 0);
  }

  if (!!impl.listener_)
  {
    cancel(impl, ec);
    if (ec)
    {
      return native_handle_type(0, 0);
    }
  }

  impl.accept_queues_.reset();

  return native_handle_type(
      impl.parameters_.release(),
      impl.listener_.release());
}

asio::error_code apple_nw_acceptor_service_base::cancel(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  ec = asio::error::operation_not_supported;
  return ec;
}

asio::error_code apple_nw_acceptor_service_base::listen(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    int backlog, asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  if (impl.listener_)
  {
    ec = asio::error::invalid_argument;
    return ec;
  }

  impl.listener_.reset(nw_listener_create(impl.parameters_));

  nw_listener_set_new_connection_limit(impl.listener_,
      backlog == socket_base::max_listen_connections
        ? NW_LISTENER_INFINITE_CONNECTION_LIMIT : backlog);

  nw_listener_set_queue(impl.listener_,
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  impl.accept_queues_.reset(new accept_queues);

  typedef apple_nw_sync_result<void> sync_result_type;
  sync_result_type result;
  __block sync_result_type* result_ptr = &result;
  __block nw_listener_t listener = impl.listener_.get();

  nw_retain(impl.listener_);
  nw_listener_set_state_changed_handler(impl.listener_,
      ^(nw_listener_state_t state, nw_error_t error)
      {
        switch (state)
        {
        case nw_listener_state_waiting:
          if (!error) break;
          // Fallthrough.
        case nw_listener_state_invalid:
        case nw_listener_state_failed:
        case nw_listener_state_ready:
        case nw_listener_state_cancelled:
          if (result_ptr)
          {
            result_ptr->set(error);
            result_ptr = 0;
          }
          if (state == nw_listener_state_cancelled)
            nw_release(listener);
          break;
        default:
          break;
        }
      });

  __block scheduler* scheduler_ptr = &scheduler_;
  __block std::weak_ptr<accept_queues> weak_queues = impl.accept_queues_;

  nw_listener_set_new_connection_handler(impl.listener_,
      ^(nw_connection_t connection)
      {
        if (connection)
        {
          if (std::shared_ptr<accept_queues> queues = weak_queues.lock())
          {
            std::unique_lock<std::mutex> lock(queues->mutex_);

            if (queues->unclaimed_connections_.size() > 64)
            {
              nw_connection_cancel(connection);
            }
            else if (queues->pending_sync_ != 0)
            {
              nw_retain(connection);
              queues->pending_sync_->set(asio::error_code(),
                  apple_nw_ptr<nw_connection_t>(connection));
              queues->pending_sync_ = 0;
            }
            else if (!queues->pending_async_.empty())
            {
              nw_retain(connection);
              apple_nw_async_op<apple_nw_ptr<nw_connection_t> >* op =
                queues->pending_async_.front();
              queues->pending_async_.pop();
              op->set(asio::error_code(),
                  apple_nw_ptr<nw_connection_t>(connection));
              scheduler_ptr->post_deferred_completion(op);
            }
            else
            {
              nw_retain(connection);
              queues->unclaimed_connections_.push_back(
                  apple_nw_ptr<nw_connection_t>(connection));
            }
          }
          else
          {
            nw_connection_cancel(connection);
          }
        }
      });

  nw_listener_start(impl.listener_);

  return result.get(ec);
}

asio::error_code apple_nw_acceptor_service_base::do_open(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    apple_nw_ptr<nw_parameters_t> parameters, asio::error_code& ec)
{
  if (is_open(impl))
  {
    ec = asio::error::already_open;
    return ec;
  }

  if (!parameters)
  {
    ec = asio::error::invalid_argument;
    return ec;
  }

  impl.parameters_ =
    ASIO_MOVE_CAST(apple_nw_ptr<nw_parameters_t>)(parameters);

  ec = asio::error_code();
  return ec;
}

asio::error_code apple_nw_acceptor_service_base::do_assign(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    const apple_nw_acceptor_service_base::native_handle_type& native_acceptor,
    asio::error_code& ec)
{
  if (is_open(impl))
  {
    ec = asio::error::already_open;
    return ec;
  }

  impl.parameters_.reset(native_acceptor.parameters);
  impl.listener_.reset(native_acceptor.listener);

  nw_listener_set_queue(impl.listener_,
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  impl.accept_queues_.reset(new accept_queues);

  ec = asio::error_code();
  return ec;
}

apple_nw_ptr<nw_endpoint_t>
apple_nw_acceptor_service_base::do_get_local_endpoint(
    const base_implementation_type& impl,
    asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  apple_nw_ptr<nw_endpoint_t> endpoint(
      nw_parameters_copy_local_endpoint(impl.parameters_));

  if (!endpoint)
  {
    ec = asio::error::invalid_argument;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  if (!!impl.listener_)
  {
    if (unsigned short port = nw_listener_get_port(impl.listener_))
    {
      switch (nw_endpoint_get_type(endpoint))
      {
      case nw_endpoint_type_address:
        {
          const socket_addr_type* address = nw_endpoint_get_address(endpoint);
          switch (address->sa_family)
          {
          case ASIO_OS_DEF(AF_INET):
            {
              sockaddr_in4_type addr_v4;
              std::memcpy(&addr_v4, address, sizeof(addr_v4));
              addr_v4.sin_port = socket_ops::host_to_network_short(port);
              endpoint.reset(
                  nw_endpoint_create_address(
                    static_cast<socket_addr_type*>(
                      static_cast<void*>(&addr_v4))));
              break;
            }
          case ASIO_OS_DEF(AF_INET6):
            {
              sockaddr_in6_type addr_v6;
              std::memcpy(&addr_v6, address, sizeof(addr_v6));
              addr_v6.sin6_port = socket_ops::host_to_network_short(port);
              endpoint.reset(
                  nw_endpoint_create_address(
                    static_cast<socket_addr_type*>(
                      static_cast<void*>(&addr_v6))));
              break;
            }
          default:
            break;
          }
          break;
        }
      case nw_endpoint_type_host:
        {
          endpoint.reset(
              nw_endpoint_create_host(
                nw_endpoint_get_hostname(endpoint),
                std::to_string(port).c_str()));
          break;
        }
      default:
        break;
      }
    }
  }

  ec = asio::error_code();
  return endpoint;
}

asio::error_code apple_nw_acceptor_service_base::do_set_option(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    const void* option, void (*set_fn)(const void*,
      nw_parameters_t, nw_listener_t, asio::error_code&),
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  set_fn(option, impl.parameters_, impl.listener_, ec);
  return ec;
}

asio::error_code apple_nw_acceptor_service_base::do_get_option(
    const apple_nw_acceptor_service_base::base_implementation_type& impl,
    void* option, void (*get_fn)(void*, nw_parameters_t,
      nw_listener_t, asio::error_code&),
    asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  get_fn(option, impl.parameters_, impl.listener_, ec);
  return ec;
}

asio::error_code apple_nw_acceptor_service_base::do_bind(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    apple_nw_ptr<nw_endpoint_t> endpoint, asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  if (!endpoint)
  {
    ec = asio::error::invalid_argument;
    return ec;
  }

  nw_parameters_set_local_endpoint(impl.parameters_, endpoint);

  ec = asio::error_code();
  return ec;
}

apple_nw_ptr<nw_connection_t> apple_nw_acceptor_service_base::do_accept(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return apple_nw_ptr<nw_connection_t>();
  }

  if (!impl.listener_ || !impl.accept_queues_)
  {
    ec = asio::error::invalid_argument;
    return apple_nw_ptr<nw_connection_t>();
  }

  std::unique_lock<std::mutex> lock(impl.accept_queues_->mutex_);

  if (!impl.accept_queues_->unclaimed_connections_.empty())
  {
    apple_nw_ptr<nw_connection_t> new_connection(
        ASIO_MOVE_CAST(apple_nw_ptr<nw_connection_t>)(
          impl.accept_queues_->unclaimed_connections_.front()));
    impl.accept_queues_->unclaimed_connections_.pop_front();
    return new_connection;
  }

  apple_nw_sync_result<apple_nw_ptr<nw_connection_t> > result;
  impl.accept_queues_->pending_sync_ = &result;

  lock.unlock();

  return result.get(ec);
}

void apple_nw_acceptor_service_base::start_accept_op(
    apple_nw_acceptor_service_base::base_implementation_type& impl,
    apple_nw_async_op<apple_nw_ptr<nw_connection_t> >* op,
    bool is_continuation, bool peer_is_open)
{
  if (!is_open(impl))
  {
    op->set(asio::error::bad_descriptor,
        apple_nw_ptr<nw_connection_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!impl.listener_ || !impl.accept_queues_)
  {
    op->set(asio::error::invalid_argument,
        apple_nw_ptr<nw_connection_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
  }

  if (peer_is_open)
  {
    op->set(asio::error::already_open,
        apple_nw_ptr<nw_connection_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  std::unique_lock<std::mutex> lock(impl.accept_queues_->mutex_);

  if (!impl.accept_queues_->unclaimed_connections_.empty())
  {
    op->set(asio::error_code(),
        apple_nw_ptr<nw_connection_t>(
          ASIO_MOVE_CAST(apple_nw_ptr<nw_connection_t>)(
            impl.accept_queues_->unclaimed_connections_.front())));
    impl.accept_queues_->unclaimed_connections_.pop_front();
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  scheduler_.work_started();
  impl.accept_queues_->pending_async_.push(op);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_IMPL_APPLE_NW_ACCEPTOR_SERVICE_BASE_IPP
