//
// win_iocp_provider.hpp
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

#ifndef ASIO_DETAIL_WIN_IOCP_PROVIDER_HPP
#define ASIO_DETAIL_WIN_IOCP_PROVIDER_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32) // This provider is only supported on Win32

#include "asio/completion_context_locker.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/service_provider.hpp"
#include "asio/detail/demuxer_thread_pool.hpp"
#include "asio/detail/dgram_socket_service.hpp"
#include "asio/detail/stream_socket_service.hpp"

namespace asio {
namespace detail {

class win_iocp_provider
  : public service_provider,
    public demuxer_service,
    public completion_context_locker,
    public dgram_socket_service,
    public stream_socket_service
{
public:
  // Constructor.
  win_iocp_provider();

  // Destructor.
  virtual ~win_iocp_provider();

  // Return the service interface corresponding to the given type.
  virtual service* do_get_service(const service_type_id& service_type);

  // Run the demuxer's event processing loop.
  virtual void run();

  // Interrupt the demuxer's event processing loop.
  virtual void interrupt();

  // Reset the demuxer in preparation for a subsequent run invocation.
  virtual void reset();

  // Add a task to the demuxer.
  virtual void add_task(demuxer_task& task, void* arg);

  // Notify the demuxer that an operation has started.
  virtual void operation_started();

  // Notify the demuxer that an operation has completed.
  virtual void operation_completed(const completion_handler& handler,
      completion_context& context, bool allow_nested_delivery);

  // Notify the demuxer of an operation that started and finished immediately.
  virtual void operation_immediate(const completion_handler& handler,
      completion_context& context, bool allow_nested_delivery);

  // Callback function when a completion context has been acquired.
  virtual void completion_context_acquired(void* arg) throw ();

  // Register a new dgram socket with the service. This should be called only
  // after the socket has been opened.
  virtual void register_dgram_socket(dgram_socket& socket);

  // Remove a dgram socket registration from the service. This should be
  // called immediately before the socket is closed.
  virtual void deregister_dgram_socket(dgram_socket& socket);

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  virtual void async_dgram_socket_sendto(dgram_socket& socket,
      const void* data, size_t length, const socket_address& destination,
      const sendto_handler& handler, completion_context& context);

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_address obejct must both be valid for the lifetime of the
  // asynchronous operation.
  virtual void async_dgram_socket_recvfrom(dgram_socket& socket, void* data,
      size_t max_length, socket_address& sender_address,
      const recvfrom_handler& handler, completion_context& context);

  // Register a new stream socket with the service. This should be called only
  // after the socket has been opened, i.e. after an accept or just before a
  // connect.
  virtual void register_stream_socket(stream_socket& socket);

  // Remove a stream socket registration from the service. This should be
  // called immediately before the socket is closed.
  virtual void deregister_stream_socket(stream_socket& socket);

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  virtual void async_stream_socket_send(stream_socket& socket,
      const void* data, size_t length, const send_handler& handler,
      completion_context& context);

  // Start an asynchronous send that will not return until all of the data has
  // been sent or an error occurs. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  virtual void async_stream_socket_send_n(stream_socket& socket,
      const void* data, size_t length, const send_n_handler& handler,
      completion_context& context);

  // Start an asynchronous receive. The buffer for the data being received must
  // be valid for the lifetime of the asynchronous operation.
  virtual void async_stream_socket_recv(stream_socket& socket, void* data,
      size_t max_length, const recv_handler& handler,
      completion_context& context);

  // Start an asynchronous receive that will not return until the specified
  // number of bytes has been received or an error occurs. The buffer for the
  // data being received must be valid for the lifetime of the asynchronous
  // operation.
  virtual void async_stream_socket_recv_n(stream_socket& socket, void* data,
      size_t length, const recv_n_handler& handler,
      completion_context& context);

private:
  // The IO completion port used for queueing operations.
  struct iocp_holder
  {
    HANDLE handle;
    iocp_holder(HANDLE h) : handle(h) {}
    ~iocp_holder() { ::CloseHandle(handle); }
  } iocp_;

  // The number of operations that have not yet completed.
  long outstanding_operations_;

  // Flag to indicate whether the event loop has been interrupted.
  long interrupted_;

  // Thread pool to keep track of threads currently inside a run invocation.
  demuxer_thread_pool thread_pool_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_PROVIDER_HPP
