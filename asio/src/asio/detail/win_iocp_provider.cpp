//
// win_iocp_provider.cpp
// ~~~~~~~~~~~~~~~~~~~~~
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

#if defined(_WIN32) // This provider is only supported on Win32

#include "asio/detail/win_iocp_provider.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service_unavailable.hpp"
#include "asio/socket_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

namespace asio {
namespace detail {

win_iocp_provider::
win_iocp_provider()
  : iocp_(::CreateIoCompletionPort(INVALID_HANDLE_VALUE, 0, 0, 0)),
    outstanding_operations_(0),
    interrupted_(0),
    thread_pool_()
{
  WSADATA wsa_data;
  ::WSAStartup(MAKEWORD(2, 0), &wsa_data);
}

win_iocp_provider::
~win_iocp_provider()
{
  ::WSACleanup();
}

service*
win_iocp_provider::
do_get_service(
    const service_type_id& service_type)
{
  if (service_type == demuxer_service::id
      || service_type == dgram_socket_service::id
      || service_type == stream_socket_service::id)
    return this;
  return 0;
}

namespace
{
  // Structure used as the base type for all operations.
  struct operation : public OVERLAPPED
  {
    completion_context* context_;
    bool context_acquired_;

    operation()
    {
      ::ZeroMemory(static_cast<OVERLAPPED*>(this), sizeof(OVERLAPPED));
    }

    virtual void do_completion(DWORD last_error, size_t bytes_transferred) = 0;
  };
} // namespace

void
win_iocp_provider::
run()
{
  if (::InterlockedExchangeAdd(&outstanding_operations_, 0) == 0)
    return;

  thread_pool_.add_current_thread();

  for (;;)
  {
    // Get the next operation from the queue.
    DWORD bytes_transferred = 0;
    DWORD_PTR completion_key = 0;
    LPOVERLAPPED overlapped = 0;
    ::SetLastError(0);
    ::GetQueuedCompletionStatus(iocp_.handle, &bytes_transferred,
        &completion_key, &overlapped, INFINITE);
    DWORD last_error = ::GetLastError();

    if (overlapped)
    {
      // Dispatch the operation.
      operation* op = static_cast<operation*>(overlapped);
      if (!op->context_acquired_ && !try_acquire(*op->context_))
      {
        acquire(*op->context_, op);
      }
      else
      {
        op->context_acquired_ = true;
        completion_context* context = op->context_;
        op->do_completion(last_error, bytes_transferred);
        release(*context);
        if (::InterlockedDecrement(&outstanding_operations_) == 0)
          interrupt();
      }
    }
    else
    {
      // The interrupted_ flag is always checked to ensure that any leftover
      // interrupts from a previous run invocation are ignored.
      if (::InterlockedExchangeAdd(&interrupted_, 0) != 0)
      {
        // Wake up next thread that is blocked on GetQueuedCompletionStatus.
        ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0);
        break;
      }
    }
  }

  thread_pool_.remove_current_thread();
}

void
win_iocp_provider::
interrupt()
{
  if (::InterlockedExchange(&interrupted_, 1) == 0)
    ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, 0);
}

void
win_iocp_provider::
reset()
{
  ::InterlockedExchange(&interrupted_, 0);
}

void
win_iocp_provider::
add_task(
    demuxer_task& task,
    void* arg)
{
}

void
win_iocp_provider::
operation_started()
{
  ::InterlockedIncrement(&outstanding_operations_);
}

namespace
{
  struct completion_operation : public operation
  {
    demuxer::completion_handler handler_;

    virtual void do_completion(DWORD, size_t)
    {
      do_upcall(handler_);
      delete this;
    }

    static void do_upcall(const demuxer::completion_handler& handler)
    {
      try
      {
        handler();
      }
      catch (...)
      {
      }
    }
  };
} // namespace

void
win_iocp_provider::
operation_completed(
    const completion_handler& handler,
    completion_context& context,
    bool allow_nested_delivery)
{
  if (try_acquire(context))
  {
    if (allow_nested_delivery && thread_pool_.current_thread_is_member())
    {
      completion_operation::do_upcall(handler);
      release(context);
      if (::InterlockedDecrement(&outstanding_operations_) == 0)
        interrupt();
    }
    else
    {
      completion_operation* op = new completion_operation;
      op->handler_ = handler;
      op->context_ = &context;
      op->context_acquired_ = true;
      ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op);
    }
  }
  else
  {
    completion_operation* op = new completion_operation;
    op->handler_ = handler;
    op->context_ = &context;
    op->context_acquired_ = false;
    acquire(context, op);
  }
}

void
win_iocp_provider::
operation_immediate(
    const completion_handler& handler,
    completion_context& context,
    bool allow_nested_delivery)
{
  operation_started();
  operation_completed(handler, context, allow_nested_delivery);
}

void
win_iocp_provider::
completion_context_acquired(
    void* arg)
  throw ()
{
  operation* op = static_cast<operation*>(arg);
  op->context_acquired_ = true;
  ::PostQueuedCompletionStatus(iocp_.handle, 0, 0, op);
}
 
void
win_iocp_provider::
do_dgram_socket_create(
    dgram_socket_service::impl_type& impl,
    const socket_address& address)
{
  socket_holder sock(socket_ops::socket(address.family(), SOCK_DGRAM,
        IPPROTO_UDP));
  if (sock.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  int reuse = 1;
  socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
      sizeof(reuse));

  if (socket_ops::bind(sock.get(), address.native_address(),
        address.native_size()) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  impl = sock.release();
}

void
win_iocp_provider::
do_dgram_socket_destroy(
    dgram_socket_service::impl_type& impl)
{
  if (impl != dgram_socket_service::invalid_impl)
  {
    socket_ops::close(impl);
    impl = dgram_socket_service::invalid_impl;
  }
}

namespace
{
  struct sendto_operation : public operation
  {
    dgram_socket_service::sendto_handler handler_;

    virtual void do_completion(DWORD last_error, size_t bytes_transferred)
    {
      socket_error error(last_error);
      do_upcall(handler_, error, bytes_transferred);
      delete this;
    }

    static void do_upcall(const dgram_socket_service::sendto_handler& handler,
        const socket_error& error, size_t bytes_transferred)
    {
      try
      {
        handler(error, bytes_transferred);
      }
      catch (...)
      {
      }
    }
  };
} // namespace

void
win_iocp_provider::
do_dgram_socket_async_sendto(
    dgram_socket_service::impl_type& impl,
    const void* data,
    size_t length,
    const socket_address& destination,
    const sendto_handler& handler,
    completion_context& context)
{
  sendto_operation* sendto_op = new sendto_operation;
  sendto_op->handler_ = handler;
  sendto_op->context_ = &context;
  sendto_op->context_acquired_ = false;

  operation_started();

  WSABUF buf;
  buf.len = length;
  buf.buf = static_cast<char*>(const_cast<void*>(data));
  DWORD bytes_transferred = 0;

  int result = ::WSASendTo(impl, &buf, 1, &bytes_transferred, 0,
      destination.native_address(), destination.native_size(), sendto_op, 0);
  DWORD last_error = ::WSAGetLastError();

  if (result != 0 && last_error != WSA_IO_PENDING)
  {
    delete sendto_op;
    socket_error error(last_error);
    operation_completed(
        boost::bind(&sendto_operation::do_upcall, handler, error,
          bytes_transferred), context, false);
  }
}

namespace
{
  struct recvfrom_operation : public operation
  {
    dgram_socket_service::recvfrom_handler handler_;
    socket_address* sender_address_;
    int sender_address_size_;

    virtual void do_completion(DWORD last_error, size_t bytes_transferred)
    {
      sender_address_->native_size(sender_address_size_);
      socket_error error(last_error);
      do_upcall(handler_, error, bytes_transferred);
      delete this;
    }

    static void do_upcall(
        const dgram_socket_service::recvfrom_handler& handler,
        const socket_error& error, size_t bytes_transferred)
    {
      try
      {
        handler(error, bytes_transferred);
      }
      catch (...)
      {
      }
    }
  };
} // namespace

void
win_iocp_provider::
do_dgram_socket_async_recvfrom(
    dgram_socket_service::impl_type& impl,
    void* data,
    size_t max_length,
    socket_address& sender_address,
    const recvfrom_handler& handler,
    completion_context& context)
{
  recvfrom_operation* recvfrom_op = new recvfrom_operation;
  recvfrom_op->handler_ = handler;
  recvfrom_op->sender_address_ = &sender_address;
  recvfrom_op->sender_address_size_ = sender_address.native_size();
  recvfrom_op->context_ = &context;
  recvfrom_op->context_acquired_ = false;

  operation_started();

  WSABUF buf;
  buf.len = max_length;
  buf.buf = static_cast<char*>(data);
  DWORD bytes_transferred = 0;
  DWORD flags = 0;

  int result = ::WSARecvFrom(impl, &buf, 1, &bytes_transferred, &flags,
      sender_address.native_address(), &recvfrom_op->sender_address_size_,
      recvfrom_op, 0);
  DWORD last_error = ::WSAGetLastError();

  if (result != 0 && last_error != WSA_IO_PENDING)
  {
    delete recvfrom_op;
    socket_error error(last_error);
    operation_completed(
        boost::bind(&recvfrom_operation::do_upcall, handler, error,
          bytes_transferred), context, false);
  }
}

void
win_iocp_provider::
do_stream_socket_create(
    stream_socket_service::impl_type& impl,
    stream_socket_service::impl_type new_impl)
{
  impl = new_impl;
}

void
win_iocp_provider::
do_stream_socket_destroy(
    stream_socket_service::impl_type& impl)
{
  if (impl != stream_socket_service::invalid_impl)
  {
    socket_ops::close(impl);
    impl = stream_socket_service::invalid_impl;
  }
}

void
win_iocp_provider::
do_stream_socket_async_send(
    stream_socket_service::impl_type& impl,
    const void* data,
    size_t length,
    const send_handler& handler,
    completion_context& context)
{
}

void
win_iocp_provider::
do_stream_socket_async_send_n(
    stream_socket_service::impl_type& impl,
    const void* data,
    size_t length,
    const send_n_handler& handler,
    completion_context& context)
{
}

void
win_iocp_provider::
do_stream_socket_async_recv(
    stream_socket_service::impl_type& impl,
    void* data,
    size_t max_length,
    const recv_handler& handler,
    completion_context& context)
{
}

void
win_iocp_provider::
do_stream_socket_async_recv_n(
    stream_socket_service::impl_type& impl,
    void* data,
    size_t length,
    const recv_n_handler& handler,
    completion_context& context)
{
}

} // namespace detail
} // namespace asio

#endif // !defined(_WIN32)
