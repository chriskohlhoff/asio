//
// reactive_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/shared_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/error_handler.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_base.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"

namespace asio {
namespace detail {

template <typename IO_Service, typename Reactor>
class reactive_socket_service
{
public:
  // The native type of a socket.
  typedef socket_type native_type;

  // The implementation type of the socket.
  class implementation_type
    : private noncopyable
  {
  private:
    // Only this service will have access to the internal values.
    friend class reactive_socket_service<IO_Service, Reactor>;

    // The native socket representation.
    socket_type socket_;

    enum
    {
      user_set_non_blocking = 1, // The user wants a non-blocking socket.
      internal_non_blocking = 2 // The socket has been set non-blocking.
    };

    // Flags indicating the current state of the socket.
    unsigned char flags_;
  };

  // The maximum number of buffers to support in a single operation.
  enum { max_buffers = 16 };

  // Constructor.
  reactive_socket_service(IO_Service& io_service)
    : io_service_(io_service),
      reactor_(io_service.get_service(service_factory<Reactor>()))
  {
  }

  // Get the io_service associated with the service.
  IO_Service& io_service()
  {
    return io_service_;
  }

  // Construct a new socket implementation.
  void construct(implementation_type& impl)
  {
    impl.socket_ = invalid_socket;
    impl.flags_ = 0;
  }

  // Destroy a socket implementation.
  void destroy(implementation_type& impl)
  {
    close(impl, asio::ignore_error());
  }

  // Open a new socket implementation.
  template <typename Protocol, typename Error_Handler>
  void open(implementation_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    close(impl, asio::ignore_error());

    socket_holder sock(socket_ops::socket(protocol.family(),
          protocol.type(), protocol.protocol()));
    if (sock.get() == invalid_socket)
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    if (int err = reactor_.register_descriptor(sock.get()))
    {
      error_handler(asio::error(err));
      return;
    }

    impl.socket_ = sock.release();
    impl.flags_ = 0;
  }

  // Open a new socket implementation from a native socket.
  template <typename Error_Handler>
  void open(implementation_type& impl, const native_type& native_socket,
      Error_Handler error_handler)
  {
    close(impl, asio::ignore_error());

    if (int err = reactor_.register_descriptor(native_socket))
    {
      error_handler(asio::error(err));
      return;
    }

    impl.socket_ = native_socket;
    impl.flags_ = 0;
  }

  // Destroy a socket implementation.
  template <typename Error_Handler>
  void close(implementation_type& impl, Error_Handler error_handler)
  {
    if (impl.socket_ != invalid_socket)
    {
      reactor_.close_descriptor(impl.socket_);

      if (impl.flags_ & implementation_type::internal_non_blocking)
      {
        ioctl_arg_type non_blocking = 0;
        socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking);
        impl.flags_ &= ~implementation_type::internal_non_blocking;
      }

      if (socket_ops::close(impl.socket_) == socket_error_retval)
        error_handler(asio::error(socket_ops::get_error()));
      else
        impl.socket_ = invalid_socket;
    }
  }

  // Get the native socket representation.
  native_type native(implementation_type& impl)
  {
    return impl.socket_;
  }

  // Bind the socket to the specified local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void bind(implementation_type& impl, const Endpoint& endpoint,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl.socket_, endpoint.data(),
          endpoint.size()) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Place the socket into the state where it will listen for new connections.
  template <typename Error_Handler>
  void listen(implementation_type& impl, int backlog,
      Error_Handler error_handler)
  {
    if (backlog == 0)
      backlog = SOMAXCONN;

    if (socket_ops::listen(impl.socket_, backlog) == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(implementation_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    if (socket_ops::setsockopt(impl.socket_, option.level(), option.name(),
          option.data(), option.size()))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const implementation_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    size_t size = option.size();
    if (socket_ops::getsockopt(impl.socket_, option.level(), option.name(),
          option.data(), &size))
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Perform an IO control command on the socket.
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(implementation_type& impl, IO_Control_Command& command,
      Error_Handler error_handler)
  {
    if (command.name() == static_cast<int>(FIONBIO))
    {
      if (command.get())
        impl.flags_ |= implementation_type::user_set_non_blocking;
      else
        impl.flags_ &= ~implementation_type::user_set_non_blocking;
    }
    else
    {
      if (socket_ops::ioctl(impl.socket_, command.name(),
            static_cast<ioctl_arg_type*>(command.data())))
        error_handler(asio::error(socket_ops::get_error()));
    }
  }

  // Get the local endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_local_endpoint(const implementation_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.size();
    if (socket_ops::getsockname(impl.socket_, endpoint.data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.size(addr_len);
  }

  // Get the remote endpoint.
  template <typename Endpoint, typename Error_Handler>
  void get_remote_endpoint(const implementation_type& impl, Endpoint& endpoint,
      Error_Handler error_handler) const
  {
    socket_addr_len_type addr_len = endpoint.size();
    if (socket_ops::getpeername(impl.socket_, endpoint.data(), &addr_len))
    {
      error_handler(asio::error(socket_ops::get_error()));
      return;
    }

    endpoint.size(addr_len);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(implementation_type& impl, socket_base::shutdown_type what,
      Error_Handler error_handler)
  {
    if (socket_ops::shutdown(impl.socket_, what) != 0)
      error_handler(asio::error(socket_ops::get_error()));
  }

  // Send the given data to the peer.
  template <typename Const_Buffers, typename Error_Handler>
  size_t send(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    // Copy buffers into array.
    socket_ops::bufs bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      asio::const_buffer buffer(*iter);
      bufs[i].size = asio::buffer_size(buffer);
      bufs[i].data = const_cast<void*>(
          asio::buffer_cast<const void*>(buffer));
    }

    // Send the data.
    for (;;)
    {
      // Try to complete the operation without blocking.
      int bytes_sent = socket_ops::send(impl.socket_, bufs, i, flags);
      int error = socket_ops::get_error();

      // Check if operation succeeded.
      if (bytes_sent >= 0)
        return bytes_sent;

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (error != asio::error::would_block
            && error != asio::error::try_again))
      {
        error_handler(asio::error(error));
        return 0;
      }

      // Wait for socket to become ready.
      if (socket_ops::poll_write(impl.socket_) < 0)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return 0;
      }
    }
  }

  template <typename Const_Buffers, typename Handler>
  class send_handler
  {
  public:
    send_handler(socket_type socket, IO_Service& io_service,
        const Const_Buffers& buffers, socket_base::message_flags flags,
        Handler handler)
      : socket_(socket),
        io_service_(io_service),
        work_(io_service),
        buffers_(buffers),
        flags_(flags),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error, 0));
        return true;
      }

      // Copy buffers into array.
      socket_ops::bufs bufs[max_buffers];
      typename Const_Buffers::const_iterator iter = buffers_.begin();
      typename Const_Buffers::const_iterator end = buffers_.end();
      size_t i = 0;
      for (; iter != end && i < max_buffers; ++iter, ++i)
      {
        asio::const_buffer buffer(*iter);
        bufs[i].size = asio::buffer_size(buffer);
        bufs[i].data = const_cast<void*>(
            asio::buffer_cast<const void*>(buffer));
      }

      // Send the data.
      int bytes = socket_ops::send(socket_, bufs, i, flags_);
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);

      // Check if we need to run the operation again.
      if (error == asio::error::would_block
          || error == asio::error::try_again)
        return false;

      io_service_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      return true;
    }

  private:
    socket_type socket_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Const_Buffers buffers_;
    socket_base::message_flags flags_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Handler>
  void async_send(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl.socket_ == invalid_socket)
    {
      asio::error error(asio::error::bad_descriptor);
      io_service_.post(bind_handler(handler, error, 0));
    }
    else
    {
      // Make socket non-blocking.
      if (!(impl.flags_ & implementation_type::internal_non_blocking))
      {
        ioctl_arg_type non_blocking = 1;
        if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
        {
          asio::error error(socket_ops::get_error());
          io_service_.post(bind_handler(handler, error, 0));
          return;
        }
        impl.flags_ |= implementation_type::internal_non_blocking;
      }

      reactor_.start_write_op(impl.socket_,
          send_handler<Const_Buffers, Handler>(
            impl.socket_, io_service_, buffers, flags, handler));
    }
  }

  // Send a datagram to the specified endpoint. Returns the number of bytes
  // sent.
  template <typename Const_Buffers, typename Endpoint, typename Error_Handler>
  size_t send_to(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const Endpoint& destination,
      Error_Handler error_handler)
  {
    // Copy buffers into array.
    socket_ops::bufs bufs[max_buffers];
    typename Const_Buffers::const_iterator iter = buffers.begin();
    typename Const_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      asio::const_buffer buffer(*iter);
      bufs[i].size = asio::buffer_size(buffer);
      bufs[i].data = const_cast<void*>(
          asio::buffer_cast<const void*>(buffer));
    }

    // Send the data.
    for (;;)
    {
      // Try to complete the operation without blocking.
      int bytes_sent = socket_ops::sendto(impl.socket_, bufs, i, flags,
          destination.data(), destination.size());
      int error = socket_ops::get_error();

      // Check if operation succeeded.
      if (bytes_sent >= 0)
        return bytes_sent;

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (error != asio::error::would_block
            && error != asio::error::try_again))
      {
        error_handler(asio::error(error));
        return 0;
      }

      // Wait for socket to become ready.
      if (socket_ops::poll_write(impl.socket_) < 0)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return 0;
      }
    }
  }

  template <typename Const_Buffers, typename Endpoint, typename Handler>
  class send_to_handler
  {
  public:
    send_to_handler(socket_type socket, IO_Service& io_service,
        const Const_Buffers& buffers, socket_base::message_flags flags,
        const Endpoint& endpoint, Handler handler)
      : socket_(socket),
        io_service_(io_service),
        work_(io_service),
        buffers_(buffers),
        flags_(flags),
        destination_(endpoint),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error, 0));
        return true;
      }

      // Copy buffers into array.
      socket_ops::bufs bufs[max_buffers];
      typename Const_Buffers::const_iterator iter = buffers_.begin();
      typename Const_Buffers::const_iterator end = buffers_.end();
      size_t i = 0;
      for (; iter != end && i < max_buffers; ++iter, ++i)
      {
        asio::const_buffer buffer(*iter);
        bufs[i].size = asio::buffer_size(buffer);
        bufs[i].data = const_cast<void*>(
            asio::buffer_cast<const void*>(buffer));
      }

      // Send the data.
      int bytes = socket_ops::sendto(socket_, bufs, i, flags_,
          destination_.data(), destination_.size());
      asio::error error(bytes < 0
          ? socket_ops::get_error() : asio::error::success);

      // Check if we need to run the operation again.
      if (error == asio::error::would_block
          || error == asio::error::try_again)
        return false;

      io_service_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      return true;
    }

  private:
    socket_type socket_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Const_Buffers buffers_;
    socket_base::message_flags flags_;
    Endpoint destination_;
    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Const_Buffers, typename Endpoint, typename Handler>
  void async_send_to(implementation_type& impl, const Const_Buffers& buffers,
      socket_base::message_flags flags, const Endpoint& destination,
      Handler handler)
  {
    if (impl.socket_ == invalid_socket)
    {
      asio::error error(asio::error::bad_descriptor);
      io_service_.post(bind_handler(handler, error, 0));
    }
    else
    {
      // Make socket non-blocking.
      if (!(impl.flags_ & implementation_type::internal_non_blocking))
      {
        ioctl_arg_type non_blocking = 1;
        if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
        {
          asio::error error(socket_ops::get_error());
          io_service_.post(bind_handler(handler, error, 0));
          return;
        }
        impl.flags_ |= implementation_type::internal_non_blocking;
      }

      reactor_.start_write_op(impl.socket_,
          send_to_handler<Const_Buffers, Endpoint, Handler>(
            impl.socket_, io_service_, buffers, flags, destination, handler));
    }
  }

  // Receive some data from the peer. Returns the number of bytes received or
  // 0 if the connection was closed cleanly.
  template <typename Mutable_Buffers, typename Error_Handler>
  size_t receive(implementation_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Error_Handler error_handler)
  {
    // Copy buffers into array.
    socket_ops::bufs bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      asio::mutable_buffer buffer(*iter);
      bufs[i].size = asio::buffer_size(buffer);
      bufs[i].data = asio::buffer_cast<void*>(buffer);
    }

    // Receive some data.
    for (;;)
    {
      // Try to complete the operation without blocking.
      int bytes_recvd = socket_ops::recv(impl.socket_, bufs, i, flags);
      int error = socket_ops::get_error();

      // Check if operation succeeded.
      if (bytes_recvd > 0)
        return bytes_recvd;

      // Check for EOF.
      if (bytes_recvd == 0)
      {
        error_handler(asio::error(asio::error::eof));
        return 0;
      }

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (error != asio::error::would_block
            && error != asio::error::try_again))
      {
        error_handler(asio::error(error));
        return 0;
      }

      // Wait for socket to become ready.
      if (socket_ops::poll_read(impl.socket_) < 0)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return 0;
      }
    }
  }

  template <typename Mutable_Buffers, typename Handler>
  class receive_handler
  {
  public:
    receive_handler(socket_type socket, IO_Service& io_service,
        const Mutable_Buffers& buffers, socket_base::message_flags flags,
        Handler handler)
      : socket_(socket),
        io_service_(io_service),
        work_(io_service),
        buffers_(buffers),
        flags_(flags),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error, 0));
        return true;
      }

      // Copy buffers into array.
      socket_ops::bufs bufs[max_buffers];
      typename Mutable_Buffers::const_iterator iter = buffers_.begin();
      typename Mutable_Buffers::const_iterator end = buffers_.end();
      size_t i = 0;
      for (; iter != end && i < max_buffers; ++iter, ++i)
      {
        asio::mutable_buffer buffer(*iter);
        bufs[i].size = asio::buffer_size(buffer);
        bufs[i].data = asio::buffer_cast<void*>(buffer);
      }

      // Receive some data.
      int bytes = socket_ops::recv(socket_, bufs, i, flags_);
      int error_code = asio::error::success;
      if (bytes < 0)
        error_code = socket_ops::get_error();
      else if (bytes == 0)
        error_code = asio::error::eof;
      asio::error error(error_code);

      // Check if we need to run the operation again.
      if (error == asio::error::would_block
          || error == asio::error::try_again)
        return false;

      io_service_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      return true;
    }

  private:
    socket_type socket_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Mutable_Buffers buffers_;
    socket_base::message_flags flags_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received
  // must be valid for the lifetime of the asynchronous operation.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(implementation_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Handler handler)
  {
    if (impl.socket_ == invalid_socket)
    {
      asio::error error(asio::error::bad_descriptor);
      io_service_.post(bind_handler(handler, error, 0));
    }
    else
    {
      // Make socket non-blocking.
      if (!(impl.flags_ & implementation_type::internal_non_blocking))
      {
        ioctl_arg_type non_blocking = 1;
        if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
        {
          asio::error error(socket_ops::get_error());
          io_service_.post(bind_handler(handler, error, 0));
          return;
        }
        impl.flags_ |= implementation_type::internal_non_blocking;
      }

      if (flags & socket_base::message_out_of_band)
      {
        reactor_.start_except_op(impl.socket_,
            receive_handler<Mutable_Buffers, Handler>(
              impl.socket_, io_service_, buffers, flags, handler));
      }
      else
      {
        reactor_.start_read_op(impl.socket_,
            receive_handler<Mutable_Buffers, Handler>(
              impl.socket_, io_service_, buffers, flags, handler));
      }
    }
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename Mutable_Buffers, typename Endpoint, typename Error_Handler>
  size_t receive_from(implementation_type& impl, const Mutable_Buffers& buffers,
      socket_base::message_flags flags, Endpoint& sender_endpoint,
      Error_Handler error_handler)
  {
    // Copy buffers into array.
    socket_ops::bufs bufs[max_buffers];
    typename Mutable_Buffers::const_iterator iter = buffers.begin();
    typename Mutable_Buffers::const_iterator end = buffers.end();
    size_t i = 0;
    for (; iter != end && i < max_buffers; ++iter, ++i)
    {
      asio::mutable_buffer buffer(*iter);
      bufs[i].size = asio::buffer_size(buffer);
      bufs[i].data = asio::buffer_cast<void*>(buffer);
    }

    // Receive some data.
    for (;;)
    {
      // Try to complete the operation without blocking.
      socket_addr_len_type addr_len = sender_endpoint.size();
      int bytes_recvd = socket_ops::recvfrom(impl.socket_, bufs, i, flags,
          sender_endpoint.data(), &addr_len);
      int error = socket_ops::get_error();

      // Check if operation succeeded.
      if (bytes_recvd > 0)
      {
        sender_endpoint.size(addr_len);
        return bytes_recvd;
      }

      // Check for EOF.
      if (bytes_recvd == 0)
      {
        error_handler(asio::error(asio::error::eof));
        return 0;
      }

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (error != asio::error::would_block
            && error != asio::error::try_again))
      {
        error_handler(asio::error(error));
        return 0;
      }

      // Wait for socket to become ready.
      if (socket_ops::poll_read(impl.socket_) < 0)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return 0;
      }
    }
  }

  template <typename Mutable_Buffers, typename Endpoint, typename Handler>
  class receive_from_handler
  {
  public:
    receive_from_handler(socket_type socket, IO_Service& io_service,
        const Mutable_Buffers& buffers, socket_base::message_flags flags,
        Endpoint& endpoint, Handler handler)
      : socket_(socket),
        io_service_(io_service),
        work_(io_service),
        buffers_(buffers),
        flags_(flags),
        sender_endpoint_(endpoint),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error, 0));
        return true;
      }

      // Copy buffers into array.
      socket_ops::bufs bufs[max_buffers];
      typename Mutable_Buffers::const_iterator iter = buffers_.begin();
      typename Mutable_Buffers::const_iterator end = buffers_.end();
      size_t i = 0;
      for (; iter != end && i < max_buffers; ++iter, ++i)
      {
        asio::mutable_buffer buffer(*iter);
        bufs[i].size = asio::buffer_size(buffer);
        bufs[i].data = asio::buffer_cast<void*>(buffer);
      }

      // Receive some data.
      socket_addr_len_type addr_len = sender_endpoint_.size();
      int bytes = socket_ops::recvfrom(socket_, bufs, i, flags_,
          sender_endpoint_.data(), &addr_len);
      int error_code = asio::error::success;
      if (bytes < 0)
        error_code = socket_ops::get_error();
      else if (bytes == 0)
        error_code = asio::error::eof;
      asio::error error(error_code);

      // Check if we need to run the operation again.
      if (error == asio::error::would_block
          || error == asio::error::try_again)
        return false;

      sender_endpoint_.size(addr_len);
      io_service_.post(bind_handler(handler_, error, bytes < 0 ? 0 : bytes));
      return true;
    }

  private:
    socket_type socket_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Mutable_Buffers buffers_;
    socket_base::message_flags flags_;
    Endpoint& sender_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_endpoint object must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Mutable_Buffers, typename Endpoint, typename Handler>
  void async_receive_from(implementation_type& impl,
      const Mutable_Buffers& buffers, socket_base::message_flags flags,
      Endpoint& sender_endpoint, Handler handler)
  {
    if (impl.socket_ == invalid_socket)
    {
      asio::error error(asio::error::bad_descriptor);
      io_service_.post(bind_handler(handler, error, 0));
    }
    else
    {
      // Make socket non-blocking.
      if (!(impl.flags_ & implementation_type::internal_non_blocking))
      {
        ioctl_arg_type non_blocking = 1;
        if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
        {
          asio::error error(socket_ops::get_error());
          io_service_.post(bind_handler(handler, error, 0));
          return;
        }
        impl.flags_ |= implementation_type::internal_non_blocking;
      }

      reactor_.start_read_op(impl.socket_,
          receive_from_handler<Mutable_Buffers, Endpoint, Handler>(
            impl.socket_, io_service_, buffers, flags,
            sender_endpoint, handler));
    }
  }

  // Accept a new connection.
  template <typename Socket, typename Error_Handler>
  void accept(implementation_type& impl, Socket& peer,
      Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.native() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    // Accept a socket.
    for (;;)
    {
      // Try to complete the operation without blocking.
      socket_holder new_socket(socket_ops::accept(impl.socket_, 0, 0));
      int error = socket_ops::get_error();

      // Check if operation succeeded.
      if (new_socket.get() >= 0)
      {
        asio::error temp_error;
        peer.open(new_socket.get(), asio::assign_error(temp_error));
        if (temp_error)
          error_handler(temp_error);
        else
          new_socket.release();
        return;
      }

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (error != asio::error::would_block
            && error != asio::error::try_again))
      {
        error_handler(asio::error(error));
        return;
      }

      // Wait for socket to become ready.
      if (socket_ops::poll_read(impl.socket_) < 0)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return;
      }
    }
  }

  // Accept a new connection.
  template <typename Socket, typename Endpoint, typename Error_Handler>
  void accept_endpoint(implementation_type& impl, Socket& peer,
      Endpoint& peer_endpoint, Error_Handler error_handler)
  {
    // We cannot accept a socket that is already open.
    if (peer.native() != invalid_socket)
    {
      error_handler(asio::error(asio::error::already_connected));
      return;
    }

    if (int err = socket_ops::get_error())
    {
      error_handler(asio::error(err));
      return;
    }

    // Accept a socket.
    for (;;)
    {
      // Try to complete the operation without blocking.
      socket_addr_len_type addr_len = peer_endpoint.size();
      socket_holder new_socket(socket_ops::accept(
            impl.socket_, peer_endpoint.data(), &addr_len));
      int error = socket_ops::get_error();

      // Check if operation succeeded.
      if (new_socket.get() >= 0)
      {
        peer_endpoint.size(addr_len);
        asio::error temp_error;
        peer.open(new_socket.get(), asio::assign_error(temp_error));
        if (temp_error)
          error_handler(temp_error);
        else
          new_socket.release();
        return;
      }

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (error != asio::error::would_block
            && error != asio::error::try_again))
      {
        error_handler(asio::error(error));
        return;
      }

      // Wait for socket to become ready.
      if (socket_ops::poll_read(impl.socket_) < 0)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return;
      }
    }
  }

  template <typename Socket, typename Handler>
  class accept_handler
  {
  public:
    accept_handler(socket_type socket, IO_Service& io_service, Socket& peer,
        Handler handler)
      : socket_(socket),
        io_service_(io_service),
        work_(io_service),
        peer_(peer),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error));
        return true;
      }

      // Accept the waiting connection.
      socket_holder new_socket(socket_ops::accept(socket_, 0, 0));
      asio::error error(new_socket.get() == invalid_socket
          ? socket_ops::get_error() : asio::error::success);

      // Check if we need to run the operation again.
      if (error == asio::error::would_block
          || error == asio::error::try_again)
        return false;

      // Transfer ownership of the new socket to the peer object.
      if (!error)
      {
        peer_.open(new_socket.get(), asio::assign_error(error));
        if (!error)
          new_socket.release();
      }

      io_service_.post(bind_handler(handler_, error));
      return true;
    }

  private:
    socket_type socket_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Socket& peer_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer object must be valid until the
  // accept's handler is invoked.
  template <typename Socket, typename Handler>
  void async_accept(implementation_type& impl, Socket& peer, Handler handler)
  {
    if (impl.socket_ == invalid_socket)
    {
      asio::error error(asio::error::bad_descriptor);
      io_service_.post(bind_handler(handler, error));
    }
    else if (peer.native() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      io_service_.post(bind_handler(handler, error));
    }
    else
    {
      // Make socket non-blocking.
      if (!(impl.flags_ & implementation_type::internal_non_blocking))
      {
        ioctl_arg_type non_blocking = 1;
        if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
        {
          asio::error error(socket_ops::get_error());
          io_service_.post(bind_handler(handler, error));
          return;
        }
        impl.flags_ |= implementation_type::internal_non_blocking;
      }

      reactor_.start_read_op(impl.socket_,
          accept_handler<Socket, Handler>(
            impl.socket_, io_service_, peer, handler));
    }
  }

  template <typename Socket, typename Endpoint, typename Handler>
  class accept_endp_handler
  {
  public:
    accept_endp_handler(socket_type socket, IO_Service& io_service,
        Socket& peer, Endpoint& peer_endpoint, Handler handler)
      : socket_(socket),
        io_service_(io_service),
        work_(io_service),
        peer_(peer),
        peer_endpoint_(peer_endpoint),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error));
        return true;
      }

      // Accept the waiting connection.
      socket_addr_len_type addr_len = peer_endpoint_.size();
      socket_holder new_socket(socket_ops::accept(
            socket_, peer_endpoint_.data(), &addr_len));
      asio::error error(new_socket.get() == invalid_socket
          ? socket_ops::get_error() : asio::error::success);

      // Check if we need to run the operation again.
      if (error == asio::error::would_block
          || error == asio::error::try_again)
        return false;

      // Transfer ownership of the new socket to the peer object.
      if (!error)
      {
        peer_endpoint_.size(addr_len);
        peer_.open(new_socket.get(), asio::assign_error(error));
        if (!error)
          new_socket.release();
      }

      io_service_.post(bind_handler(handler_, error));
      return true;
    }

  private:
    socket_type socket_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Socket& peer_;
    Endpoint& peer_endpoint_;
    Handler handler_;
  };

  // Start an asynchronous accept. The peer and peer_endpoint objects
  // must be valid until the accept's handler is invoked.
  template <typename Socket, typename Endpoint, typename Handler>
  void async_accept_endpoint(implementation_type& impl, Socket& peer,
      Endpoint& peer_endpoint, Handler handler)
  {
    if (impl.socket_ == invalid_socket)
    {
      asio::error error(asio::error::bad_descriptor);
      io_service_.post(bind_handler(handler, error));
    }
    else if (peer.native() != invalid_socket)
    {
      asio::error error(asio::error::already_connected);
      io_service_.post(bind_handler(handler, error));
    }
    else
    {
      // Make socket non-blocking.
      if (!(impl.flags_ & implementation_type::internal_non_blocking))
      {
        ioctl_arg_type non_blocking = 1;
        if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
        {
          asio::error error(socket_ops::get_error());
          io_service_.post(bind_handler(handler, error));
          return;
        }
        impl.flags_ |= implementation_type::internal_non_blocking;
      }

      reactor_.start_read_op(impl.socket_,
          accept_endp_handler<Socket, Endpoint, Handler>(
            impl.socket_, io_service_, peer, peer_endpoint, handler));
    }
  }

  // Connect the socket to the specified endpoint.
  template <typename Endpoint, typename Error_Handler>
  void connect(implementation_type& impl, const Endpoint& peer_endpoint,
      Error_Handler error_handler)
  {
    // Open the socket if it is not already open.
    if (impl.socket_ == invalid_socket)
    {
      // Get the flags used to create the new socket.
      int family = peer_endpoint.protocol().family();
      int type = peer_endpoint.protocol().type();
      int proto = peer_endpoint.protocol().protocol();

      // Create a new socket.
      impl.socket_ = socket_ops::socket(family, type, proto);
      if (impl.socket_ == invalid_socket)
      {
        error_handler(asio::error(socket_ops::get_error()));
        return;
      }

      // Register the socket with the reactor.
      if (int err = reactor_.register_descriptor(impl.socket_))
      {
        socket_ops::close(impl.socket_);
        error_handler(asio::error(err));
        return;
      }
    }
    else if (impl.flags_ & implementation_type::internal_non_blocking)
    {
      // Mark the socket as blocking while we perform the connect.
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
      {
        error_handler(asio::error(socket_ops::get_error()));
        return;
      }
      impl.flags_ &= ~implementation_type::internal_non_blocking;
    }

    // Perform the connect operation.
    int result = socket_ops::connect(impl.socket_,
        peer_endpoint.data(), peer_endpoint.size());
    if (result == socket_error_retval)
      error_handler(asio::error(socket_ops::get_error()));
  }

  template <typename Handler>
  class connect_handler
  {
  public:
    connect_handler(socket_type socket, boost::shared_ptr<bool> completed,
        IO_Service& io_service, Reactor& reactor, Handler handler)
      : socket_(socket),
        completed_(completed),
        io_service_(io_service),
        work_(io_service),
        reactor_(reactor),
        handler_(handler)
    {
    }

    bool operator()(int result)
    {
      // Check whether a handler has already been called for the connection.
      // If it has, then we don't want to do anything in this handler.
      if (*completed_)
        return true;

      // Cancel the other reactor operation for the connection.
      *completed_ = true;
      reactor_.enqueue_cancel_ops_unlocked(socket_);

      // Check whether the operation was successful.
      if (result != 0)
      {
        asio::error error(result);
        io_service_.post(bind_handler(handler_, error));
        return true;
      }

      // Get the error code from the connect operation.
      int connect_error = 0;
      size_t connect_error_len = sizeof(connect_error);
      if (socket_ops::getsockopt(socket_, SOL_SOCKET, SO_ERROR,
            &connect_error, &connect_error_len) == socket_error_retval)
      {
        asio::error error(socket_ops::get_error());
        io_service_.post(bind_handler(handler_, error));
        return true;
      }

      // If connection failed then post the handler with the error code.
      if (connect_error)
      {
        asio::error error(connect_error);
        io_service_.post(bind_handler(handler_, error));
        return true;
      }

      // Post the result of the successful connection operation.
      asio::error error(asio::error::success);
      io_service_.post(bind_handler(handler_, error));

      return true;
    }

  private:
    socket_type socket_;
    boost::shared_ptr<bool> completed_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Reactor& reactor_;
    Handler handler_;
  };

  // Start an asynchronous connect.
  template <typename Endpoint, typename Handler>
  void async_connect(implementation_type& impl, const Endpoint& peer_endpoint,
      Handler handler)
  {
    // Open the socket if it is not already open.
    if (impl.socket_ == invalid_socket)
    {
      // Get the flags used to create the new socket.
      int family = peer_endpoint.protocol().family();
      int type = peer_endpoint.protocol().type();
      int proto = peer_endpoint.protocol().protocol();

      // Create a new socket.
      impl.socket_ = socket_ops::socket(family, type, proto);
      if (impl.socket_ == invalid_socket)
      {
        asio::error error(socket_ops::get_error());
        io_service_.post(bind_handler(handler, error));
        return;
      }

      // Register the socket with the reactor.
      if (int err = reactor_.register_descriptor(impl.socket_))
      {
        socket_ops::close(impl.socket_);
        asio::error error(err);
        io_service_.post(bind_handler(handler, error));
        return;
      }
    }

    // Make socket non-blocking.
    if (!(impl.flags_ & implementation_type::internal_non_blocking))
    {
      ioctl_arg_type non_blocking = 1;
      if (socket_ops::ioctl(impl.socket_, FIONBIO, &non_blocking))
      {
        asio::error error(socket_ops::get_error());
        io_service_.post(bind_handler(handler, error));
        return;
      }
      impl.flags_ |= implementation_type::internal_non_blocking;
    }

    // Start the connect operation. The socket is already marked as non-blocking
    // so the connection will take place asynchronously.
    if (socket_ops::connect(impl.socket_, peer_endpoint.data(),
          peer_endpoint.size()) == 0)
    {
      // The connect operation has finished successfully so we need to post the
      // handler immediately.
      asio::error error(asio::error::success);
      io_service_.post(bind_handler(handler, error));
    }
    else if (socket_ops::get_error() == asio::error::in_progress
        || socket_ops::get_error() == asio::error::would_block)
    {
      // The connection is happening in the background, and we need to wait
      // until the socket becomes writeable.
      boost::shared_ptr<bool> completed(new bool(false));
      reactor_.start_write_and_except_ops(impl.socket_,
          connect_handler<Handler>(
            impl.socket_, completed, io_service_, reactor_, handler));
    }
    else
    {
      // The connect operation has failed, so post the handler immediately.
      asio::error error(socket_ops::get_error());
      io_service_.post(bind_handler(handler, error));
    }
  }

private:
  // The io_service used for dispatching handlers.
  IO_Service& io_service_;

  // The selector that performs event demultiplexing for the provider.
  Reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_SOCKET_SERVICE_HPP
