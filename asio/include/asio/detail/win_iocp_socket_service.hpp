//
// win_iocp_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

// This service is only supported on Win32 (NT4 and later).
#if defined(BOOST_WINDOWS)
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/buffer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_demuxer_service.hpp"

namespace asio {
namespace detail {

template <typename Allocator>
class win_iocp_socket_service
{
public:
  struct noop_deleter { void operator()(void*) {} };
  typedef boost::shared_ptr<void> shared_cancel_token_type;
  typedef boost::weak_ptr<void> weak_cancel_token_type;

  // The native type of the socket. We use a custom class here rather than just
  // SOCKET to workaround the broken Windows support for cancellation. MSDN says
  // that when you call closesocket any outstanding WSARecv or WSASend
  // operations will complete with the error ERROR_OPERATION_ABORTED. In
  // practice they complete with ERROR_NETNAME_DELETED, which means you can't
  // tell the difference between a local cancellation and the socket being
  // hard-closed by the peer.
  class impl_type
  {
  public:
    // Default constructor.
    impl_type()
      : socket_(invalid_socket),
        cancel_token_(static_cast<void*>(0), noop_deleter())
    {
    }

    // Construct from socket type.
    explicit impl_type(socket_type s)
      : socket_(s),
        cancel_token_(static_cast<void*>(0), noop_deleter())
    {
    }

    // Copy constructor.
    impl_type(const impl_type& other)
      : socket_(other.socket_),
        cancel_token_(other.cancel_token_)
    {
    }

    // Assignment operator.
    impl_type& operator=(const impl_type& other)
    {
      socket_ = other.socket_;
      cancel_token_ = other.cancel_token_;
      return *this;
    }

    // Assign from socket type.
    impl_type& operator=(socket_type s)
    {
      cancel_token_.reset(static_cast<void*>(0), noop_deleter());
      socket_ = s;
      return *this;
    }

    // Convert to socket type.
    operator socket_type() const
    {
      return socket_;
    }

  private:
    socket_type socket_;
    friend class win_iocp_socket_service<Allocator>;
    shared_cancel_token_type cancel_token_;
  };

  static impl_type null_impl_;

  // The demuxer type for this service.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

  // The type of the reactor used for connect operations.
  typedef detail::select_reactor<true> reactor_type;

  // The maximum number of buffers to support in a single operation.
  enum { max_buffers = 16 };

  // Constructor. This socket service can only work if the demuxer is
  // using the win_iocp_demuxer_service. By using this type as the parameter we
  // will cause a compile error if this is not the case.
  win_iocp_socket_service(
      demuxer_type& demuxer)
    : demuxer_(demuxer),
      demuxer_service_(demuxer.get_service(
          service_factory<win_iocp_demuxer_service>())),
      reactor_(0)
  {
  }

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Return a null socket implementation.
  static impl_type null()
  {
    return null_impl_;
  }

  // Open a new socket implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    socket_holder sock(socket_ops::socket(protocol.family(), protocol.type(),
          protocol.protocol()));
    if (sock.get() == invalid_socket)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    demuxer_service_.register_socket(sock.get());

    impl = sock.release();
  }

  // Assign a new socket implementation.
  void assign(impl_type& impl, impl_type new_impl)
  {
    demuxer_service_.register_socket(new_impl);
    impl = new_impl;
  }

  // Destroy a socket implementation.
  template <typename Error_Handler>
  void close(impl_type& impl, Error_Handler error_handler)
  {
    if (impl != null())
    {
      // Check if the reactor was created, in which case we need to close the
      // socket on the reactor as well to cancel any operations that might be
      // running there.
      reactor_type* reactor = static_cast<reactor_type*>(
          ::InterlockedCompareExchangePointer(
            reinterpret_cast<void**>(&reactor_), 0, 0));
      if (reactor)
        reactor->close_descriptor(impl);

      if (socket_ops::close(impl) == socket_error_retval)
        error_handler(asio::error(socket_ops::get_error()));
      else
        impl = null();
    }
  }

  // Bind the socket to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(impl_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, endpoint.data(),
          endpoint.size()) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Place the socket into the state where it will listen for new connections.
  template <typename Error_Handler>
  void listen(impl_type& impl, int backlog, Error_Handler error_handler)
  {
    if (backlog == 0)
      backlog = SOMAXCONN;

    if (socket_ops::listen(impl, backlog) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(impl_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    if (socket_ops::setsockopt(impl, option.level(), option.name(),
          option.data(), option.size()))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const impl_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(impl_type& impl, IO_Control_Command& command,
      Error_Handler error_handler)
  {
    if (socket_ops::ioctl(impl, command.name(),
          static_cast<ioctl_arg_type*>(command.data())))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.size();
    if (socket_ops::getsockname(impl, endpoint.data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.size(addr_len);
  }

  // Get the remote endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(const impl_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.size();
    if (socket_ops::getpeername(impl, endpoint.data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.size(addr_len);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(impl_type& impl, socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    if (socket_ops::shutdown(impl, what) != 0)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Send the given data to the peer. Returns the number of bytes sent or
  // 0 if the connection was closed cleanly.
  template <typename Const_Buffers, typename Error_Handler>
  size_t send(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = const_cast<char*>(
          asio::buffer_cast<const char*>(*iter));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASend(impl, bufs, i, &bytes_transferred, flags, 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      if (last_error == ERROR_NETNAME_DELETED)
        last_error = WSAECONNRESET;
      error_handler(asio::error(last_error));
      return 0;
    }

    return bytes_transferred;
  }

  template <typename Handler>
  class send_operation
    : public win_iocp_operation
  {
  public:
    send_operation(demuxer_type& demuxer,
        weak_cancel_token_type cancel_token, Handler handler)
      : win_iocp_operation(&send_operation<Handler>::do_completion_impl),
        work_(demuxer),
        cancel_token_(cancel_token),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        DWORD last_error, size_t bytes_transferred)
    {
      std::auto_ptr<send_operation<Handler> > h(
          static_cast<send_operation<Handler>*>(op));

      // Map ERROR_NETNAME_DELETED to more useful error.
      if (last_error == ERROR_NETNAME_DELETED)
      {
        if (h->cancel_token_.expired())
          last_error = ERROR_OPERATION_ABORTED;
        else
          last_error = WSAECONNRESET;
      }

      asio::error error(last_error);
      h->handler_(error, bytes_transferred);
    }

    typename demuxer_type::work work_;
    weak_cancel_token_type cancel_token_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_send(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    std::auto_ptr<send_operation<Handler> > op(
        new send_operation<Handler>(demuxer_, impl.cancel_token_, handler));

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = const_cast<char*>(
          asio::buffer_cast<const char*>(*iter));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASend(impl, bufs, i,
        &bytes_transferred, flags, op.get(), 0);
    DWORD last_error = ::WSAGetLastError();

    // Check if the operation completed immediately.
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
    }
    else
    {
      op.release();
    }
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename Const_Buffers, typename Endpoint, typename Error_Handler>
  size_t send_to(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const Endpoint& destination,
      Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = const_cast<char*>(
          asio::buffer_cast<const char*>(*iter));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASendTo(impl, bufs, i, &bytes_transferred, flags,
        destination.data(), destination.size(), 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      error_handler(asio::error(last_error));
      return 0;
    }

    return bytes_transferred;
  }

  template <typename Handler>
  class send_to_operation
    : public win_iocp_operation
  {
  public:
    send_to_operation(demuxer_type& demuxer, Handler handler)
      : win_iocp_operation(&send_to_operation<Handler>::do_completion_impl),
        work_(demuxer),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        DWORD last_error, size_t bytes_transferred)
    {
      std::auto_ptr<send_to_operation<Handler> > h(
          static_cast<send_to_operation<Handler>*>(op));
      asio::error error(last_error);
      h->handler_(error, bytes_transferred);
    }

    typename demuxer_type::work work_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Endpoint, typename Handler>
  void async_send_to(impl_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const Endpoint& destination,
      Handler handler)
  {
    std::auto_ptr<send_to_operation<Handler> > op(
        new send_to_operation<Handler>(demuxer_, handler));

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = const_cast<char*>(
          asio::buffer_cast<const char*>(*iter));
    }

    // Send the data.
    DWORD bytes_transferred = 0;
    int result = ::WSASendTo(impl, bufs, i, &bytes_transferred, flags,
        destination.data(), destination.size(), op.get(), 0);
    DWORD last_error = ::WSAGetLastError();

    // Check if the operation completed immediately.
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
    }
    else
    {
      op.release();
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly.
  template <typename Mutable_Buffers, typename Error_Handler>
  size_t receive(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = asio::buffer_cast<char*>(*iter);
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int result = ::WSARecv(impl, bufs, i,
        &bytes_transferred, &recv_flags, 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      if (last_error == ERROR_NETNAME_DELETED)
        last_error = WSAECONNRESET;
      error_handler(asio::error(last_error));
      return 0;
    }
    if (bytes_transferred == 0)
    {
      error_handler(asio::error(asio::error::eof));
      return 0;
    }

    return bytes_transferred;
  }

  template <typename Handler>
  class receive_operation
    : public win_iocp_operation
  {
  public:
    receive_operation(demuxer_type& demuxer,
        weak_cancel_token_type cancel_token, Handler handler)
      : win_iocp_operation(&receive_operation<Handler>::do_completion_impl),
        work_(demuxer),
        cancel_token_(cancel_token),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        DWORD last_error, size_t bytes_transferred)
    {
      std::auto_ptr<receive_operation<Handler> > h(
          static_cast<receive_operation<Handler>*>(op));

      // Map ERROR_NETNAME_DELETED to more useful error.
      if (last_error == ERROR_NETNAME_DELETED)
      {
        if (h->cancel_token_.expired())
          last_error = ERROR_OPERATION_ABORTED;
        else
          last_error = WSAECONNRESET;
      }

      // Check for connection closed.
      else if (last_error == 0 && bytes_transferred == 0)
      {
        last_error = asio::error::eof;
      }

      asio::error error(last_error);
      h->handler_(error, bytes_transferred);
    }

    typename demuxer_type::work work_;
    weak_cancel_token_type cancel_token_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    std::auto_ptr<receive_operation<Handler> > op(
        new receive_operation<Handler>(demuxer_, impl.cancel_token_, handler));

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = asio::buffer_cast<char*>(*iter);
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int result = ::WSARecv(impl, bufs, i,
        &bytes_transferred, &recv_flags, op.get(), 0);
    DWORD last_error = ::WSAGetLastError();
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
    }
    else
    {
      op.release();
    }
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename Mutable_Buffers, typename Endpoint, typename Error_Handler>
  size_t receive_from(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Endpoint& sender_endpoint,
      Error_Handler error_handler)
  {
    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = asio::buffer_cast<char*>(*iter);
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int endpoint_size = sender_endpoint.size();
    int result = ::WSARecvFrom(impl, bufs, i, &bytes_transferred, &recv_flags,
        sender_endpoint.data(), &endpoint_size, 0, 0);
    if (result != 0)
    {
      DWORD last_error = ::WSAGetLastError();
      error_handler(asio::error(last_error));
      return 0;
    }
    if (bytes_transferred == 0)
    {
      error_handler(asio::error(asio::error::eof));
      return 0;
    }

    sender_endpoint.size(endpoint_size);

    return bytes_transferred;
  }

  template <typename Endpoint, typename Handler>
  class receive_from_operation
    : public win_iocp_operation
  {
  public:
    receive_from_operation(demuxer_type& demuxer, Endpoint& endpoint,
        Handler handler)
      : win_iocp_operation(
          &receive_from_operation<Endpoint, Handler>::do_completion_impl),
        endpoint_(endpoint),
        endpoint_size_(endpoint.size()),
        work_(demuxer),
        handler_(handler)
    {
    }

    int& endpoint_size()
    {
      return endpoint_size_;
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        DWORD last_error, size_t bytes_transferred)
    {
      std::auto_ptr<receive_from_operation<Endpoint, Handler> > h(
          static_cast<receive_from_operation<Endpoint, Handler>*>(op));

      // Check for connection closed.
      if (last_error == 0 && bytes_transferred == 0)
      {
        last_error = asio::error::eof;
      }

      h->endpoint_.size(h->endpoint_size_);
      asio::error error(last_error);
      h->handler_(error, bytes_transferred);
    }

    Endpoint& endpoint_;
    int endpoint_size_;
    typename demuxer_type::work work_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Mutable_Buffers, typename Endpoint, typename Handler>
  void async_receive_from(impl_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Endpoint& sender_endp, Handler handler)
  {
    std::auto_ptr<receive_from_operation<Endpoint, Handler> > op(
        new receive_from_operation<Endpoint, Handler>(
          demuxer_, sender_endp, handler));

    // Copy buffers into WSABUF array.
    ::WSABUF bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    DWORD i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      bufs[i].len = static_cast<u_long>(asio::buffer_size(*iter));
      bufs[i].buf = asio::buffer_cast<char*>(*iter);
    }

    // Receive some data.
    DWORD bytes_transferred = 0;
    DWORD recv_flags = flags;
    int result = ::WSARecvFrom(impl, bufs, i, &bytes_transferred, &recv_flags,
        sender_endp.data(), &op->endpoint_size(), op.get(), 0);
    DWORD last_error = ::WSAGetLastError();
    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
    }
    else
    {
      op.release();
    }
  }

  // Accept a new connection.
  template <typename Socket, typename Error_Handler>
  void accept(impl_type& impl, Socket& peer, Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    impl_type new_socket(socket_ops::accept(impl, 0, 0));
    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    peer.set_impl(new_socket);
  }

  // Accept a new connection.
  template <typename Socket, typename Endpoint, typename Error_Handler>
  void accept_endpoint(impl_type& impl, Socket& peer, Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.impl() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    socket_addr_len_type addr_len = peer_endpoint.size();
    impl_type new_socket(socket_ops::accept(impl,
          peer_endpoint.data(), &addr_len));
    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    peer_endpoint.size(addr_len);

    peer.set_impl(new_socket);
  }

  template <typename Socket, typename Handler>
  class accept_operation
    : public win_iocp_operation
  {
  public:
    accept_operation(demuxer_type& demuxer, impl_type& impl,
        socket_type new_socket, Socket& peer, Handler handler)
      : win_iocp_operation(
          &accept_operation<Socket, Handler>::do_completion_impl),
        demuxer_(demuxer),
        impl_(impl),
        new_socket_(new_socket),
        peer_(peer),
        work_(demuxer),
        handler_(handler)
    {
    }

    socket_type new_socket()
    {
      return new_socket_.get();
    }

    void* output_buffer()
    {
      return output_buffer_;
    }

    DWORD address_length()
    {
      return sizeof(sockaddr_storage) + 16;
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        DWORD last_error, size_t bytes_transferred)
    {
      std::auto_ptr<accept_operation<Socket, Handler> > h(
          static_cast<accept_operation<Socket, Handler>*>(op));

      // Check for connection aborted.
      if (last_error == ERROR_NETNAME_DELETED)
      {
        last_error = asio::error::connection_aborted;
      }

      // Check whether the operation was successful.
      if (last_error != 0)
      {
        asio::error error(last_error);
        h->handler_(error);
        return;
      }

      // Need to set the SO_UPDATE_ACCEPT_CONTEXT option so that getsockname
      // and getpeername will work on the accepted socket.
      DWORD update_ctx_param = h->impl_;
      if (socket_ops::setsockopt(h->new_socket_.get(), SOL_SOCKET,
            SO_UPDATE_ACCEPT_CONTEXT, &update_ctx_param, sizeof(DWORD)) != 0)
      {
        asio::error error(socket_ops::get_error());
        h->handler_(error);
        return;
      }

      // Socket was successfully connected. Transfer ownership of the socket to
      // the peer object.
      impl_type new_socket(h->new_socket_.get());
      h->peer_.set_impl(new_socket);
      h->new_socket_.release();
      asio::error error(asio::error::success);
      h->handler_(error);
    }

    demuxer_type& demuxer_;
    impl_type& impl_;
    socket_holder new_socket_;
    Socket& peer_;
    typename demuxer_type::work work_;
    unsigned char output_buffer_[(sizeof(sockaddr_storage) + 16) * 2];
    Handler handler_;
  };

  // Start an asynchronous accept. The peer object must be valid until the
  // accept's handler is invoked.
  template <typename Socket, typename Handler>
  void async_accept(impl_type& impl, Socket& peer, Handler handler)
  {
    // Check whether acceptor has been initialised.
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Check that peer socket has not already been connected.
    if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Get information about the protocol used by the socket.
    WSAPROTOCOL_INFO protocol_info;
    std::size_t protocol_info_size = sizeof(protocol_info);
    if (socket_ops::getsockopt(impl, SOL_SOCKET, SO_PROTOCOL_INFO,
          &protocol_info, &protocol_info_size) != 0)
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Create a new socket for the connection.
    socket_holder sock(socket_ops::socket(protocol_info.iAddressFamily,
          protocol_info.iSocketType, protocol_info.iProtocol));
    if (sock.get() == invalid_socket)
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Create new operation object. Ownership of new socket is transferred.
    std::auto_ptr<accept_operation<Socket, Handler> > op(
        new accept_operation<Socket, Handler>(
          demuxer_, impl, sock.get(), peer, handler));
    sock.release();

    // Accept a connection.
    DWORD bytes_read = 0;
    BOOL result = ::AcceptEx(impl, op->new_socket(), op->output_buffer(), 0,
        op->address_length(), op->address_length(), &bytes_read, op.get());
    DWORD last_error = ::WSAGetLastError();

    // Check if the operation completed immediately.
    if (!result && last_error != WSA_IO_PENDING)
    {
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error));
    }
    else
    {
      op.release();
    }
  }

  template <typename Socket, typename Endpoint, typename Handler>
  class accept_endp_operation
    : public win_iocp_operation
  {
  public:
    accept_endp_operation(demuxer_type& demuxer, impl_type& impl,
        socket_type new_socket, Socket& peer, Endpoint& peer_endpoint,
        Handler handler)
      : win_iocp_operation(&accept_endp_operation<
            Socket, Endpoint, Handler>::do_completion_impl),
        demuxer_(demuxer),
        impl_(impl),
        new_socket_(new_socket),
        peer_(peer),
        peer_endpoint_(peer_endpoint),
        work_(demuxer),
        handler_(handler)
    {
    }

    socket_type new_socket()
    {
      return new_socket_.get();
    }

    void* output_buffer()
    {
      return output_buffer_;
    }

    DWORD address_length()
    {
      return sizeof(sockaddr_storage) + 16;
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        DWORD last_error, size_t bytes_transferred)
    {
      std::auto_ptr<accept_endp_operation<Socket, Endpoint, Handler> > h(
          static_cast<accept_endp_operation<Socket, Endpoint, Handler>*>(op));

      // Check for connection aborted.
      if (last_error == ERROR_NETNAME_DELETED)
      {
        last_error = asio::error::connection_aborted;
      }

      // Check whether the operation was successful.
      if (last_error != 0)
      {
        asio::error error(last_error);
        h->handler_(error);
        return;
      }

      // Get the address of the peer.
      LPSOCKADDR local_addr = 0;
      int local_addr_length = 0;
      LPSOCKADDR remote_addr = 0;
      int remote_addr_length = 0;
      GetAcceptExSockaddrs(h->output_buffer(), 0, h->address_length(),
          h->address_length(), &local_addr, &local_addr_length, &remote_addr,
          &remote_addr_length);
      if (remote_addr_length > h->peer_endpoint_.size())
      {
        asio::error error(asio::error::invalid_argument);
        h->handler_(error);
        return;
      }
      h->peer_endpoint_.size(remote_addr_length);
      using namespace std; // For memcpy.
      memcpy(h->peer_endpoint_.data(), remote_addr, remote_addr_length);

      // Need to set the SO_UPDATE_ACCEPT_CONTEXT option so that getsockname
      // and getpeername will work on the accepted socket.
      DWORD update_ctx_param = h->impl_;
      if (socket_ops::setsockopt(h->new_socket_.get(), SOL_SOCKET,
            SO_UPDATE_ACCEPT_CONTEXT, &update_ctx_param, sizeof(DWORD)) != 0)
      {
        asio::error error(socket_ops::get_error());
        h->handler_(error);
        return;
      }

      // Socket was successfully connected. Transfer ownership of the socket to
      // the peer object.
      impl_type new_socket(h->new_socket_.get());
      h->peer_.set_impl(new_socket);
      h->new_socket_.release();
      asio::error error(asio::error::success);
      h->handler_(error);
    }

    demuxer_type& demuxer_;
    impl_type& impl_;
    socket_holder new_socket_;
    Socket& peer_;
    Endpoint& peer_endpoint_;
    typename demuxer_type::work work_;
    unsigned char output_buffer_[(sizeof(sockaddr_storage) + 16) * 2];
    Handler handler_;
  };

  // Start an asynchronous accept. The peer and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Socket, typename Endpoint, typename Handler>
  void async_accept_endpoint(impl_type& impl, Socket& peer,
      Endpoint& peer_endpoint, Handler handler)
  {
    // Check whether acceptor has been initialised.
    if (impl == null())
    {
      asio::error error(asio::error::bad_descriptor);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Check that peer socket has not already been connected.
    if (peer.impl() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Get information about the protocol used by the socket.
    WSAPROTOCOL_INFO protocol_info;
    std::size_t protocol_info_size = sizeof(protocol_info);
    if (socket_ops::getsockopt(impl, SOL_SOCKET, SO_PROTOCOL_INFO,
          &protocol_info, &protocol_info_size) != 0)
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Create a new socket for the connection.
    socket_holder sock(socket_ops::socket(protocol_info.iAddressFamily,
          protocol_info.iSocketType, protocol_info.iProtocol));
    if (sock.get() == invalid_socket)
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Create new operation object. Ownership of new socket is transferred.
    std::auto_ptr<accept_endp_operation<Socket, Endpoint, Handler> > op(
        new accept_endp_operation<Socket, Endpoint, Handler>(
          demuxer_, impl, sock.get(), peer, peer_endpoint, handler));
    sock.release();

    // Accept a connection.
    DWORD bytes_read = 0;
    BOOL result = ::AcceptEx(impl, op->new_socket(), op->output_buffer(), 0,
        op->address_length(), op->address_length(), &bytes_read, op.get());
    DWORD last_error = ::WSAGetLastError();

    // Check if the operation completed immediately.
    if (!result && last_error != WSA_IO_PENDING)
    {
      asio::error error(last_error);
      demuxer_service_.post(bind_handler(handler, error));
    }
    else
    {
      op.release();
    }
  }

  // Connect the socket to the specified endpoint.
  template <typename Endpoint, typename Error_Handler>
  void connect(impl_type& impl, const Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    // Open the socket if it is not already open.
    if (impl == invalid_socket)
    {
      // Get the flags used to create the new socket.
      int family = peer_endpoint.protocol().family();
      int type = peer_endpoint.protocol().type();
      int proto = peer_endpoint.protocol().protocol();

      // Create a new socket.
      impl = socket_ops::socket(family, type, proto);
      if (impl == invalid_socket)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return;
      }
      demuxer_service_.register_socket(impl);
    }

    // Perform the connect operation.
    int result = socket_ops::connect(impl, peer_endpoint.data(),
        peer_endpoint.size());
    if (result == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  template <typename Handler>
  class connect_handler
  {
  public:
    connect_handler(impl_type& impl, boost::shared_ptr<bool> completed,
        demuxer_type& demuxer, reactor_type& reactor, Handler handler)
      : impl_(impl),
        completed_(completed),
        demuxer_(demuxer),
        reactor_(reactor),
        work_(demuxer),
        handler_(handler)
    {
    }

    void operator()(int result)
    {
      // Check whether a handler has already been called for the connection.
      // If it has, then we don't want to do anything in this handler.
      if (*completed_)
        return;

      // Cancel the other reactor operation for the connection.
      *completed_ = true;
      reactor_.enqueue_cancel_ops_unlocked(impl_);

      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Get the error code from the connect operation.
      int connect_error = 0;
      size_t connect_error_len = sizeof(connect_error);
      if (socket_ops::getsockopt(impl_, SOL_SOCKET, SO_ERROR,
            &connect_error, &connect_error_len) == socket_error_retval)
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // If connection failed then post the handler with the error code.
      if (connect_error)
      {
        asio::error error(connect_error);
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Make the socket blocking again (the default).
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(impl_, FIONBIO, &non_blocking))
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler_, error));
        return;
      }

      // Post the result of the successful connection operation.
      asio::error error(asio::error::success);
      demuxer_.post(bind_handler(handler_, error));
    }

  private:
    impl_type& impl_;
    boost::shared_ptr<bool> completed_;
    demuxer_type& demuxer_;
    reactor_type& reactor_;
    typename demuxer_type::work work_;
    Handler handler_;
  };

  // Start an asynchronous connect.
  template <typename Endpoint, typename Handler>
  void async_connect(impl_type& impl, const Endpoint& peer_endpoint,
      Handler handler)
  {
    // Check if the reactor was already obtained from the demuxer.
    reactor_type* reactor = static_cast<reactor_type*>(
        ::InterlockedCompareExchangePointer(
          reinterpret_cast<void**>(&reactor_), 0, 0));
    if (!reactor)
    {
      reactor = &(demuxer_.get_service(service_factory<reactor_type>()));
      InterlockedExchangePointer(reinterpret_cast<void**>(&reactor_), reactor);
    }

    // Open the socket if it is not already open.
    if (impl == invalid_socket)
    {
      // Get the flags used to create the new socket.
      int family = peer_endpoint.protocol().family();
      int type = peer_endpoint.protocol().type();
      int proto = peer_endpoint.protocol().protocol();

      // Create a new socket.
      impl = socket_ops::socket(family, type, proto);
      if (impl == invalid_socket)
      {
        asio::error error(socket_ops::get_error());
        demuxer_.post(bind_handler(handler, error));
        return;
      }
      demuxer_service_.register_socket(impl);
    }

    // Mark the socket as non-blocking so that the connection will take place
    // asynchronously.
    ioctl_arg_type non_blocking = 1;
    if (socket_ops::ioctl(impl, FIONBIO, &non_blocking))
    {
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
      return;
    }

    // Start the connect operation.
    if (socket_ops::connect(impl, peer_endpoint.data(),
          peer_endpoint.size()) == 0)
    {
      // The connect operation has finished successfully so we need to post the
      // handler immediately.
      asio::error error(asio::error::success);
      demuxer_.post(bind_handler(handler, error));
    }
    else if (socket_ops::get_error() == asio::error::in_progress
        || socket_ops::get_error() == asio::error::would_block)
    {
      // The connection is happening in the background, and we need to wait
      // until the socket becomes writeable.
      boost::shared_ptr<bool> completed(new bool(false));
      reactor->start_write_and_except_ops(impl, connect_handler<Handler>(
            impl, completed, demuxer_, *reactor, handler));
    }
    else
    {
      // The connect operation has failed, so post the handler immediately.
      asio::error error(socket_ops::get_error());
      demuxer_.post(bind_handler(handler, error));
    }
  }

private:
  // The demuxer associated with the service.
  demuxer_type& demuxer_;

  // The demuxer service used for running asynchronous operations and
  // dispatching handlers.
  win_iocp_demuxer_service& demuxer_service_;

  // The reactor used for performing connect operations. This object is created
  // only if needed.
  reactor_type* reactor_;
};

template <typename Allocator>
typename win_iocp_socket_service<Allocator>::impl_type
win_iocp_socket_service<Allocator>::null_impl_;

} // namespace detail
} // namespace asio

#endif // defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0400)
#endif // defined(BOOST_WINDOWS)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_SOCKET_SERVICE_HPP
