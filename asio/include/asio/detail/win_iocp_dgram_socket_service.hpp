//
// win_iocp_dgram_socket_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#if defined(_WIN32) // This service is only supported on Win32

#include "asio/basic_demuxer.hpp"
#include "asio/service_factory.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/win_iocp_demuxer_service.hpp"

namespace asio {
namespace detail {

class win_iocp_dgram_socket_service
{
public:
  // The native type of the dgram socket. This type is dependent on the
  // underlying implementation of the socket layer.
  typedef socket_type impl_type;

  // Return a null dgram socket implementation.
  static impl_type null()
  {
    return invalid_socket;
  }

  // The demuxer type for this service.
  typedef basic_demuxer<win_iocp_demuxer_service> demuxer_type;

  // Constructor. This dgram_socket service can only work if the demuxer is
  // using the win_iocp_demuxer_service. By using this type as the parameter we
  // will cause a compile error if this is not the case.
  win_iocp_dgram_socket_service(
      demuxer_type& demuxer)
    : demuxer_(demuxer),
      demuxer_service_(demuxer.get_service(
          service_factory<win_iocp_demuxer_service>()))
  {
  }

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Open a new dgram socket implementation.
  template <typename Protocol, typename Error_Handler>
  void open(impl_type& impl, const Protocol& protocol,
      Error_Handler error_handler)
  {
    if (protocol.type() != SOCK_DGRAM)
    {
      error_handler(socket_error(socket_error::invalid_argument));
      return;
    }

    socket_holder sock(socket_ops::socket(protocol.family(), protocol.type(),
          protocol.protocol()));
    if (sock.get() == invalid_socket)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    demuxer_service_.register_socket(sock.get());

    impl = sock.release();
  }

  // Bind the dgram socket to the specified local address.
  template <typename Address, typename Error_Handler>
  void bind(impl_type& impl, const Address& address,
      Error_Handler error_handler)
  {
    if (socket_ops::bind(impl, address.native_address(),
          address.native_size()) == socket_error_retval)
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Destroy a dgram socket implementation.
  void close(impl_type& impl)
  {
    if (impl != null())
    {
      socket_ops::close(impl);
      impl = null();
    }
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(impl_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    if (socket_ops::setsockopt(impl, option.level(), option.name(),
          option.data(), option.size()))
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Set a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(impl_type& impl, Option& option, Error_Handler error_handler)
  {
    socket_len_type size = option.size();
    if (socket_ops::getsockopt(impl, option.level(), option.name(),
          option.data(), &size))
      error_handler(socket_error(socket_ops::get_error()));
  }

  // Get the local socket address.
  template <typename Address, typename Error_Handler>
  void get_local_address(impl_type& impl, Address& address,
      Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = address.native_size();
    if (socket_ops::getsockname(impl, address.native_address(), &addr_len))
    {
      error_handler(socket_error(socket_ops::get_error()));
      return;
    }

    address.native_size(addr_len);
  }

  // Send a datagram to the specified address. Returns the number of bytes
  // sent.
  template <typename Address, typename Error_Handler>
  size_t sendto(impl_type& impl, const void* data, size_t length,
      const Address& destination, Error_Handler error_handler)
  {
    int bytes_sent = socket_ops::sendto(impl, data, length, 0,
        destination.native_address(), destination.native_size());
    if (bytes_sent < 0)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return 0;
    }

    return bytes_sent;
  }

  template <typename Handler>
  class sendto_operation
    : public win_iocp_operation
  {
  public:
    sendto_operation(Handler handler)
      : win_iocp_operation(&sendto_operation<Handler>::do_completion_impl),
        handler_(handler)
    {
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      sendto_operation<Handler>* h =
        static_cast<sendto_operation<Handler>*>(op);
      socket_error error(last_error);
      try
      {
        h->handler_(error, bytes_transferred);
      }
      catch (...)
      {
      }
      demuxer_service.work_finished();
      delete h;
    }

    Handler handler_;
  };

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename Address, typename Handler>
  void async_sendto(impl_type& impl, const void* data, size_t length,
      const Address& destination, Handler handler)
  {
    sendto_operation<Handler>* sendto_op =
      new sendto_operation<Handler>(handler);

    demuxer_service_.work_started();

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
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

  // Receive a datagram with the address of the sender. Returns the number of
  // bytes received.
  template <typename Address, typename Error_Handler>
  size_t recvfrom(impl_type& impl, void* data, size_t max_length,
      Address& sender_address, Error_Handler error_handler)
  {
    socket_addr_len_type addr_len = sender_address.native_size();
    int bytes_recvd = socket_ops::recvfrom(impl, data, max_length, 0,
        sender_address.native_address(), &addr_len);
    if (bytes_recvd < 0)
    {
      error_handler(socket_error(socket_ops::get_error()));
      return 0;
    }

    sender_address.native_size(addr_len);

    return bytes_recvd;
  }

  template <typename Address, typename Handler>
  class recvfrom_operation
    : public win_iocp_operation
  {
  public:
    recvfrom_operation(Address& address, Handler handler)
      : win_iocp_operation(
          &recvfrom_operation<Address, Handler>::do_completion_impl),
        address_(address),
        address_size_(address.native_size()),
        handler_(handler)
    {
    }

    int& address_size()
    {
      return address_size_;
    }

  private:
    static void do_completion_impl(win_iocp_operation* op,
        win_iocp_demuxer_service& demuxer_service, HANDLE iocp,
        DWORD last_error, size_t bytes_transferred)
    {
      recvfrom_operation<Address, Handler>* h =
        static_cast<recvfrom_operation<Address, Handler>*>(op);
      h->address_.native_size(h->address_size_);
      socket_error error(last_error);
      try
      {
        h->handler_(error, bytes_transferred);
      }
      catch (...)
      {
      }
      demuxer_service.work_finished();
      delete h;
    }

    Address& address_;
    int address_size_;
    Handler handler_;
  };

  // Start an asynchronous receive. The buffer for the data being received and
  // the sender_address obejct must both be valid for the lifetime of the
  // asynchronous operation.
  template <typename Address, typename Handler>
  void async_recvfrom(impl_type& impl, void* data, size_t max_length,
      Address& sender_address, Handler handler)
  {
    recvfrom_operation<Address, Handler>* recvfrom_op =
      new recvfrom_operation<Address, Handler>(sender_address, handler);

    demuxer_service_.work_started();

    WSABUF buf;
    buf.len = max_length;
    buf.buf = static_cast<char*>(data);
    DWORD bytes_transferred = 0;
    DWORD flags = 0;

    int result = ::WSARecvFrom(impl, &buf, 1, &bytes_transferred, &flags,
        sender_address.native_address(), &recvfrom_op->address_size(),
        recvfrom_op, 0);
    DWORD last_error = ::WSAGetLastError();

    if (result != 0 && last_error != WSA_IO_PENDING)
    {
      delete recvfrom_op;
      socket_error error(last_error);
      demuxer_service_.post(bind_handler(handler, error, bytes_transferred));
      demuxer_service_.work_finished();
    }
  }

private:
  // The demuxer associated with the service.
  demuxer_type& demuxer_;

  // The demuxer service used for running asynchronous operations and
  // dispatching handlers.
  win_iocp_demuxer_service& demuxer_service_;
};

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_DGRAM_SOCKET_SERVICE_HPP
