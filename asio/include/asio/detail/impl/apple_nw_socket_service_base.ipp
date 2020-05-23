//
// detail/impl/apple_nw_socket_service_base.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_APPLE_NW_SOCKET_SERVICE_BASE_IPP
#define ASIO_DETAIL_IMPL_APPLE_NW_SOCKET_SERVICE_BASE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#include "asio/detail/apple_nw_socket_service_base.hpp"
#include "asio/detail/apple_nw_sync_result.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

apple_nw_socket_service_base::apple_nw_socket_service_base(
    execution_context& context)
  : scheduler_(use_service<scheduler>(context)),
    mutex_(),
    impl_list_(0),
    async_scope_()
{
}

void apple_nw_socket_service_base::base_shutdown()
{
  // Close all implementations, causing all operations to complete.
  asio::detail::mutex::scoped_lock lock(mutex_);
  base_implementation_type* impl = impl_list_;
  while (impl)
  {
    if (impl->connection_)
      nw_connection_force_cancel(impl->connection_);
    impl = impl->next_;
  }

  async_scope_.wait();
}

void apple_nw_socket_service_base::construct(
    apple_nw_socket_service_base::base_implementation_type& impl)
{
  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void apple_nw_socket_service_base::base_move_construct(
    apple_nw_socket_service_base::base_implementation_type& impl,
    apple_nw_socket_service_base::base_implementation_type& other_impl)
  ASIO_NOEXCEPT
{
  impl.parameters_.swap(other_impl.parameters_);
  other_impl.parameters_.reset();
  impl.connection_.swap(other_impl.connection_);
  other_impl.connection_.reset();
  impl.max_receive_size_ = other_impl.max_receive_size_;
  other_impl.max_receive_size_ = 0;

  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void apple_nw_socket_service_base::base_move_assign(
    apple_nw_socket_service_base::base_implementation_type& impl,
    apple_nw_socket_service_base& other_service,
    apple_nw_socket_service_base::base_implementation_type& other_impl)
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
  impl.connection_.swap(other_impl.connection_);
  other_impl.connection_.reset();
  impl.max_receive_size_ = other_impl.max_receive_size_;
  other_impl.max_receive_size_ = 0;

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

void apple_nw_socket_service_base::destroy(
    apple_nw_socket_service_base::base_implementation_type& impl)
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

asio::error_code apple_nw_socket_service_base::close(
    apple_nw_socket_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (impl.connection_)
  {
    nw_connection_cancel(impl.connection_);
  }

  impl.parameters_.reset();
  impl.connection_.reset();
  impl.max_receive_size_ = 0;

  ec = asio::error_code();
  return ec;
}

apple_nw_socket_service_base::native_handle_type
apple_nw_socket_service_base::release(
    apple_nw_socket_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    return native_handle_type(0, 0);
  }

  if (!!impl.connection_)
  {
    cancel(impl, ec);
    if (ec)
    {
      return native_handle_type(0, 0);
    }
  }

  impl.max_receive_size_ = 0;

  return native_handle_type(
      impl.parameters_.release(),
      impl.connection_.release());
}

asio::error_code apple_nw_socket_service_base::cancel(
    apple_nw_socket_service_base::base_implementation_type& impl,
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

asio::error_code apple_nw_socket_service_base::do_open(
    apple_nw_socket_service_base::base_implementation_type& impl,
    apple_nw_ptr<nw_parameters_t> parameters,
    std::size_t max_receive_size, asio::error_code& ec)
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
  impl.max_receive_size_ = max_receive_size;

  ec = asio::error_code();
  return ec;
}

asio::error_code apple_nw_socket_service_base::do_assign(
    apple_nw_socket_service_base::base_implementation_type& impl,
    const apple_nw_socket_service_base::native_handle_type& native_socket,
    std::size_t max_receive_size, asio::error_code& ec)
{
  if (is_open(impl))
  {
    ec = asio::error::already_open;
    return ec;
  }

  impl.parameters_.reset(native_socket.parameters);
  impl.connection_.reset(native_socket.connection);
  impl.max_receive_size_ = max_receive_size;

  if (!impl.parameters_)
  {
    impl.parameters_.reset(nw_connection_copy_parameters(impl.connection_));
  }

  nw_connection_set_queue(impl.connection_,
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  __block nw_connection_t connection = impl.connection_.get();

  nw_retain(impl.connection_);
  nw_connection_set_state_changed_handler(impl.connection_,
      ^(nw_connection_state_t state, nw_error_t error)
      {
        switch (state)
        {
        case nw_connection_state_waiting:
          if (!error) break;
          // Fallthrough.
        case nw_connection_state_invalid:
        case nw_connection_state_failed:
        case nw_connection_state_ready:
        case nw_connection_state_cancelled:
          if (state == nw_connection_state_cancelled)
            nw_release(connection);
          break;
        default:
          break;
        }
      });

  nw_connection_start(impl.connection_);

  ec = asio::error_code();
  return ec;
}

apple_nw_ptr<nw_endpoint_t>
apple_nw_socket_service_base::do_get_local_endpoint(
    const base_implementation_type& impl,
    asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  apple_nw_ptr<nw_endpoint_t> endpoint;
  if (!!impl.connection_)
  {
    apple_nw_ptr<nw_path_t> path(
        nw_connection_copy_current_path(impl.connection_));

    endpoint.reset(nw_path_copy_effective_local_endpoint(path));
  }
  else
  {
    endpoint.reset(nw_parameters_copy_local_endpoint(impl.parameters_));
  }

  if (!endpoint)
  {
    ec = asio::error::invalid_argument;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  ec = asio::error_code();
  return endpoint;
}

apple_nw_ptr<nw_endpoint_t>
apple_nw_socket_service_base::do_get_remote_endpoint(
    const base_implementation_type& impl,
    asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  if (!impl.connection_)
  {
    ec = asio::error::not_connected;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  apple_nw_ptr<nw_path_t> path(
      nw_connection_copy_current_path(impl.connection_));

  apple_nw_ptr<nw_endpoint_t> endpoint(
      nw_path_copy_effective_remote_endpoint(path));

  if (!endpoint)
    endpoint.reset(nw_connection_copy_endpoint(impl.connection_));

  if (!endpoint)
  {
    ec = asio::error::not_connected;
    return apple_nw_ptr<nw_endpoint_t>();
  }

  ec = asio::error_code();
  return endpoint;
}

asio::error_code apple_nw_socket_service_base::do_set_option(
    apple_nw_socket_service_base::base_implementation_type& impl,
    const void* option, void (*set_fn)(const void*,
      nw_parameters_t, nw_connection_t, asio::error_code&),
    asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  set_fn(option, impl.parameters_, impl.connection_, ec);
  return ec;
}

asio::error_code apple_nw_socket_service_base::do_get_option(
    const apple_nw_socket_service_base::base_implementation_type& impl,
    void* option, void (*get_fn)(void*, nw_parameters_t,
      nw_connection_t, asio::error_code&),
    asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  get_fn(option, impl.parameters_, impl.connection_, ec);
  return ec;
}

asio::error_code apple_nw_socket_service_base::do_bind(
    apple_nw_socket_service_base::base_implementation_type& impl,
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

asio::error_code apple_nw_socket_service_base::do_connect(
    apple_nw_socket_service_base::base_implementation_type& impl,
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

  if (!!impl.connection_)
  {
    ec = asio::error::already_connected;
    return ec;
  }

  impl.connection_.reset(nw_connection_create(endpoint, impl.parameters_));
  if (!impl.connection_)
  {
    ec = asio::error::invalid_argument;
    return ec;
  }

  nw_connection_set_queue(impl.connection_,
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  typedef apple_nw_sync_result<void> sync_result_type;
  sync_result_type result;
  __block sync_result_type* result_ptr = &result;
  __block nw_connection_t connection = impl.connection_.get();

  nw_retain(impl.connection_);
  nw_connection_set_state_changed_handler(impl.connection_,
      ^(nw_connection_state_t state, nw_error_t error)
      {
        switch (state)
        {
        case nw_connection_state_waiting:
          if (!error) break;
          // Fallthrough.
        case nw_connection_state_invalid:
        case nw_connection_state_failed:
        case nw_connection_state_ready:
        case nw_connection_state_cancelled:
          if (result_ptr)
          {
            result_ptr->set(error);
            result_ptr = 0;
          }
          if (state == nw_connection_state_cancelled)
            nw_release(connection);
          break;
        default:
          break;
        }
      });

  nw_connection_start(impl.connection_);

  return result.get(ec);
}

void apple_nw_socket_service_base::start_connect_op(
    apple_nw_socket_service_base::base_implementation_type& impl,
    apple_nw_ptr<nw_endpoint_t> endpoint, apple_nw_async_op<void>* op,
    bool is_continuation)
{
  if (!is_open(impl))
  {
    op->set(asio::error::bad_descriptor);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!endpoint)
  {
    op->set(asio::error::invalid_argument);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!!impl.connection_)
  {
    op->set(asio::error::already_connected);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  impl.connection_.reset(nw_connection_create(endpoint, impl.parameters_));
  if (!impl.connection_)
  {
    op->set(asio::error::invalid_argument);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  nw_connection_set_queue(impl.connection_,
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));

  scheduler_.work_started();
  async_scope_.work_started();

  __block scheduler* scheduler_ptr = &scheduler_;
  __block apple_nw_async_scope* scope_ptr = &async_scope_;
  __block apple_nw_async_op<void>* op_ptr = op;
  __block nw_connection_t connection = impl.connection_;

  nw_retain(impl.connection_);
  nw_connection_set_state_changed_handler(impl.connection_,
      ^(nw_connection_state_t state, nw_error_t error)
      {
        switch (state)
        {
        case nw_connection_state_waiting:
          if (!error) break;
          // Fallthrough.
        case nw_connection_state_invalid:
        case nw_connection_state_failed:
        case nw_connection_state_ready:
        case nw_connection_state_cancelled:
          if (op_ptr)
          {
            op_ptr->set(error);
            scheduler_ptr->post_deferred_completion(op_ptr);
            scope_ptr->work_finished();
            op_ptr = 0;
          }
          if (state == nw_connection_state_cancelled)
            nw_release(connection);
          break;
        default:
          break;
        }
      });

  nw_connection_start(impl.connection_);
}

asio::error_code apple_nw_socket_service_base::do_shutdown(
    apple_nw_socket_service_base::base_implementation_type& impl,
    socket_base::shutdown_type what, asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  if (what != socket_base::shutdown_send)
  {
    ec = asio::error::operation_not_supported;
    return ec;
  }

  typedef apple_nw_sync_result<void> sync_result_type;
  sync_result_type result;
  __block sync_result_type& result_ref = result;

  nw_connection_send(impl.connection_, dispatch_data_empty,
      NW_CONNECTION_FINAL_MESSAGE_CONTEXT, true,
      ^(nw_error_t error)
      {
        result_ref.set(error);
      });

  return result.get(ec);
}

std::size_t apple_nw_socket_service_base::do_send(
    apple_nw_socket_service_base::base_implementation_type& impl,
    apple_nw_ptr<dispatch_data_t> data, std::size_t data_size,
    socket_base::message_flags flags, asio::error_code& ec)
{
  if (data_size == 0)
  {
    ec = asio::error_code();
    return 0;
  }

  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  if (!impl.connection_)
  {
    ec = asio::error::not_connected;
    return 0;
  }

  if (flags)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  typedef apple_nw_sync_result<void> sync_result_type;
  sync_result_type result;
  __block sync_result_type& result_ref = result;

  nw_connection_send(impl.connection_, data,
      NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true,
      ^(nw_error_t error)
      {
        result_ref.set(error);
      });

  result.get(ec);
  return ec ? 0 : data_size;
}

void apple_nw_socket_service_base::start_send_op(
      apple_nw_socket_service_base::base_implementation_type& impl,
      apple_nw_ptr<dispatch_data_t> data, std::size_t data_size,
      socket_base::message_flags flags, apple_nw_async_op<void>* op,
      bool is_continuation)
{
  if (data_size == 0)
  {
    op->set(asio::error_code());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!is_open(impl))
  {
    op->set(asio::error::bad_descriptor);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!impl.connection_)
  {
    op->set(asio::error::not_connected);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (flags)
  {
    op->set(asio::error::operation_not_supported);
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  scheduler_.work_started();
  async_scope_.work_started();

  __block scheduler* scheduler_ptr = &scheduler_;
  __block apple_nw_async_scope* scope_ptr = &async_scope_;
  __block apple_nw_async_op<void>* op_ptr = op;

  nw_connection_send(impl.connection_, data,
      NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, true,
      ^(nw_error_t error)
      {
        op_ptr->set(error);
        scheduler_ptr->post_deferred_completion(op_ptr);
        scope_ptr->work_finished();
      });
}

apple_nw_ptr<dispatch_data_t> apple_nw_socket_service_base::do_receive(
    apple_nw_socket_service_base::base_implementation_type& impl,
    std::size_t max_size, socket_base::message_flags flags,
    asio::error_code& ec)
{
  if (max_size == 0)
  {
    ec = asio::error_code();
    return apple_nw_ptr<dispatch_data_t>();
  }

  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return apple_nw_ptr<dispatch_data_t>();
  }

  if (!impl.connection_)
  {
    ec = asio::error::not_connected;
    return apple_nw_ptr<dispatch_data_t>();
  }

  if (flags)
  {
    ec = asio::error::operation_not_supported;
    return apple_nw_ptr<dispatch_data_t>();
  }

  typedef apple_nw_sync_result<apple_nw_ptr<dispatch_data_t> > sync_result_type;
  sync_result_type result;
  __block sync_result_type& result_ref = result;

  nw_connection_receive(impl.connection_, 1,
      max_size < impl.max_receive_size_ ? impl.max_receive_size_ : max_size,
      ^(dispatch_data_t receive_data,
        nw_content_context_t context,
        bool is_complete, nw_error_t error)
      {
        if (error)
        {
          result_ref.set(error, apple_nw_ptr<dispatch_data_t>());
        }
        else if (is_complete && context != 0
            && nw_content_context_get_is_final(context))
        {
          result_ref.set(asio::error::eof,
              apple_nw_ptr<dispatch_data_t>());
        }
        else
        {
          nw_retain(receive_data);
          result_ref.set(error, apple_nw_ptr<dispatch_data_t>(receive_data));
        }
      });

  return result.get(ec);
}

void apple_nw_socket_service_base::start_receive_op(
      apple_nw_socket_service_base::base_implementation_type& impl,
      std::size_t max_size, socket_base::message_flags flags,
      apple_nw_async_op<apple_nw_ptr<dispatch_data_t> >* op,
      bool is_continuation)
{
  if (max_size == 0)
  {
    op->set(asio::error_code(),
        apple_nw_ptr<dispatch_data_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!is_open(impl))
  {
    op->set(asio::error::bad_descriptor,
        apple_nw_ptr<dispatch_data_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!impl.connection_)
  {
    op->set(asio::error::not_connected,
        apple_nw_ptr<dispatch_data_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (flags)
  {
    op->set(asio::error::operation_not_supported,
        apple_nw_ptr<dispatch_data_t>());
    scheduler_.post_immediate_completion(op, is_continuation);
    return;
  }

  scheduler_.work_started();
  async_scope_.work_started();

  __block scheduler* scheduler_ptr = &scheduler_;
  __block apple_nw_async_scope* scope_ptr = &async_scope_;
  __block apple_nw_async_op<apple_nw_ptr<dispatch_data_t> >* op_ptr = op;

  nw_connection_receive(impl.connection_, 1,
      max_size < impl.max_receive_size_ ? impl.max_receive_size_ : max_size,
      ^(dispatch_data_t receive_data,
        nw_content_context_t context,
        bool is_complete, nw_error_t error)
      {
        if (error)
        {
          op_ptr->set(error, apple_nw_ptr<dispatch_data_t>());
        }
        else if (is_complete && context != 0
            && nw_content_context_get_is_final(context))
        {
          op_ptr->set(asio::error::eof,
              apple_nw_ptr<dispatch_data_t>());
        }
        else
        {
          nw_retain(receive_data);
          op_ptr->set(error, apple_nw_ptr<dispatch_data_t>(receive_data));
        }

        scheduler_ptr->post_deferred_completion(op_ptr);
        scope_ptr->work_finished();
      });
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_HAS_APPLE_NETWORK_FRAMEWORK)

#endif // ASIO_DETAIL_IMPL_APPLE_NW_SOCKET_SERVICE_BASE_IPP
