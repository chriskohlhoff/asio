#ifndef ASIO_DETAIL_IMPL_WINRT_DSOCKET_SERVICE_BASE_IPP
#define ASIO_DETAIL_IMPL_WINRT_DSOCKET_SERVICE_BASE_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include <cstring>
#include "asio/detail/winrt_dsocket_service_base.hpp"
#include "asio/detail/winrt_async_op.hpp"
#include "asio/detail/winrt_utils.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

winrt_dsocket_service_base::winrt_dsocket_service_base(
    asio::io_service& io_service)
  : io_service_(use_service<io_service_impl>(io_service)),
    async_manager_(use_service<winrt_async_manager>(io_service)),
    mutex_(),
    impl_list_(0)
{
}

void winrt_dsocket_service_base::shutdown_service()
{
  // Close all implementations, causing all operations to complete.
  asio::detail::mutex::scoped_lock lock(mutex_);
  base_implementation_type* impl = impl_list_;
  while (impl)
  {
    asio::error_code ignored_ec;
    close(*impl, ignored_ec);
    impl = impl->next_;
  }
}

void winrt_dsocket_service_base::construct(
    winrt_dsocket_service_base::base_implementation_type& impl)
{
  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void winrt_dsocket_service_base::base_move_construct(
    winrt_dsocket_service_base::base_implementation_type& impl,
    winrt_dsocket_service_base::base_implementation_type& other_impl)
{
  impl.receive_handler_ = std::move(other_impl.receive_handler_);
  impl.socket_ = other_impl.socket_;
  other_impl.socket_ = nullptr;

  // Insert implementation into linked list of all implementations.
  asio::detail::mutex::scoped_lock lock(mutex_);
  impl.next_ = impl_list_;
  impl.prev_ = 0;
  if (impl_list_)
    impl_list_->prev_ = &impl;
  impl_list_ = &impl;
}

void winrt_dsocket_service_base::base_move_assign(
    winrt_dsocket_service_base::base_implementation_type& impl,
    winrt_dsocket_service_base& other_service,
    winrt_dsocket_service_base::base_implementation_type& other_impl)
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

  impl.receive_handler_ = std::move(other_impl.receive_handler_);
  impl.socket_ = other_impl.socket_;
  other_impl.socket_ = nullptr;

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

void winrt_dsocket_service_base::destroy(
    winrt_dsocket_service_base::base_implementation_type& impl)
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

asio::error_code winrt_dsocket_service_base::close(
    winrt_dsocket_service_base::base_implementation_type& impl,
    asio::error_code& ec)
{
  if (impl.socket_)
  {
    delete impl.socket_;
    impl.socket_ = nullptr;
  }

  ec = asio::error_code();
  return ec;
}

std::size_t winrt_dsocket_service_base::do_get_endpoint(
    const base_implementation_type& impl, bool local,
    void* addr, std::size_t addr_len, asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return addr_len;
  }

  try
  {
    std::string addr_string = winrt_utils::string(local
        ? impl.socket_->Information->LocalAddress->CanonicalName
        : impl.socket_->Information->RemoteAddress->CanonicalName);
    unsigned short port = winrt_utils::integer(local
        ? impl.socket_->Information->LocalPort
        : impl.socket_->Information->RemotePort);
    unsigned long scope = 0;

    switch (reinterpret_cast<const socket_addr_type*>(addr)->sa_family)
    {
    case ASIO_OS_DEF(AF_INET):
      if (addr_len < sizeof(sockaddr_in4_type))
      {
        ec = asio::error::invalid_argument;
        return addr_len;
      }
      else
      {
        socket_ops::inet_pton(ASIO_OS_DEF(AF_INET), addr_string.c_str(),
            &reinterpret_cast<sockaddr_in4_type*>(addr)->sin_addr, &scope, ec);
        reinterpret_cast<sockaddr_in4_type*>(addr)->sin_port
          = socket_ops::host_to_network_short(port);
        ec = asio::error_code();
        return sizeof(sockaddr_in4_type);
      }
    case ASIO_OS_DEF(AF_INET6):
      if (addr_len < sizeof(sockaddr_in6_type))
      {
        ec = asio::error::invalid_argument;
        return addr_len;
      }
      else
      {
        socket_ops::inet_pton(ASIO_OS_DEF(AF_INET6), addr_string.c_str(),
            &reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_addr, &scope, ec);
        reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_port
          = socket_ops::host_to_network_short(port);
        ec = asio::error_code();
        return sizeof(sockaddr_in6_type);
      }
    default:
      ec = asio::error::address_family_not_supported;
      return addr_len;
    }
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
    return addr_len;
  }
}
asio::error_code winrt_dsocket_service_base::do_set_option(
    winrt_dsocket_service_base::base_implementation_type& impl,
    int level, int optname, const void* optval,
    std::size_t optlen, asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  try
  {
    if ((level == ASIO_OS_DEF(IPPROTO_IP) && optname == ASIO_OS_DEF(IP_TTL))
        || (level == ASIO_OS_DEF(IPPROTO_IPV6) && optname == ASIO_OS_DEF(IPV6_UNICAST_HOPS)))
    {
      if (optlen == sizeof(int))
      {
        int value = 0;
        std::memcpy(&value, optval, optlen);
        impl.socket_->Control->OutboundUnicastHopLimit = value;
        ec = asio::error_code();
      }
      else
      {
        ec = asio::error::invalid_argument;
      }
    }
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0603)
    else if (level == ASIO_OS_DEF(SOL_SOCKET)
        && optname == ASIO_OS_DEF(SO_RCVBUF))
    {
      if (optlen == sizeof(int))
      {
        int value = 0;
        std::memcpy(&value, optval, optlen);
        impl.socket_->Control->InboundBufferSizeInBytes = value;
        ec = asio::error_code();
      }
      else
      {
        ec = asio::error::invalid_argument;
      }
    }
    else if (level == ASIO_OS_DEF(IPPROTO_IP)
        && optname == ASIO_OS_DEF(IP_DONTFRAGMENT))
    {
      if (optlen == sizeof(int))
      {
        int value = 0;
        std::memcpy(&value, optval, optlen);
        impl.socket_->Control->DontFragment = !!value;
        ec = asio::error_code();
      }
      else
      {
        ec = asio::error::invalid_argument;
      }
    }
#endif
    else
    {
      ec = asio::error::invalid_argument;
    }
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
  }

  return ec;
}

void winrt_dsocket_service_base::do_get_option(
    const winrt_dsocket_service_base::base_implementation_type& impl,
    int level, int optname, void* optval,
    std::size_t* optlen, asio::error_code& ec) const
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return;
  }

  try
  {
    if ((level == ASIO_OS_DEF(IPPROTO_IP) && optname == ASIO_OS_DEF(IP_TTL))
        || (level == ASIO_OS_DEF(IPPROTO_IPV6) && optname == ASIO_OS_DEF(IPV6_UNICAST_HOPS)))
    {
      if (*optlen >= sizeof(int))
      {
        int value = impl.socket_->Control->OutboundUnicastHopLimit;
        std::memcpy(optval, &value, sizeof(int));
        *optlen = sizeof(int);
        ec = asio::error_code();
      }
      else
      {
        ec = asio::error::invalid_argument;
      }
    }
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0603)
    else if (level == ASIO_OS_DEF(SOL_SOCKET)
        && optname == ASIO_OS_DEF(SO_RCVBUF))
    {
      if (*optlen >= sizeof(int))
      {
        int value = impl.socket_->Control->InboundBufferSizeInBytes;
        std::memcpy(optval, &value, sizeof(int));
        *optlen = sizeof(int);
        ec = asio::error_code();
      }
      else
      {
        ec = asio::error::invalid_argument;
      }
    }
    else if (level == ASIO_OS_DEF(IPPROTO_IP)
        && optname == ASIO_OS_DEF(IP_DONTFRAGMENT))
    {
      if (*optlen == sizeof(int))
      {
        int value = impl.socket_->Control->DontFragment ? 1 : 0;
        std::memcpy(optval, &value, sizeof(int));
        *optlen = sizeof(int);
        ec = asio::error_code();
      }
      else
      {
        ec = asio::error::invalid_argument;
      }
    }
#endif
    else
    {
      ec = asio::error::invalid_argument;
    }
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
  }
}

asio::error_code winrt_dsocket_service_base::do_connect(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const void* addr, asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  char addr_string[max_addr_v6_str_len];
  unsigned short port;
  switch (reinterpret_cast<const socket_addr_type*>(addr)->sa_family)
  {
  case ASIO_OS_DEF(AF_INET):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET),
        &reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_addr,
        addr_string, sizeof(addr_string), 0, ec);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_port);
    break;
  case ASIO_OS_DEF(AF_INET6):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET6),
        &reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_addr,
        addr_string, sizeof(addr_string), 0, ec);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_port);
    break;
  default:
    ec = asio::error::address_family_not_supported;
    return ec;
  }

  if (!ec) try
  {
    async_manager_.sync(impl.socket_->ConnectAsync(
          ref new Windows::Networking::HostName(
            winrt_utils::string(addr_string)),
          winrt_utils::string(port)), ec);
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
  }

  return ec;
}

void winrt_dsocket_service_base::start_connect_op(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const void* addr, winrt_async_op<void>* op, bool is_continuation)
{
  if (!is_open(impl))
  {
    op->ec_ = asio::error::bad_descriptor;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  char addr_string[max_addr_v6_str_len];
  unsigned short port = 0;
  switch (reinterpret_cast<const socket_addr_type*>(addr)->sa_family)
  {
  case ASIO_OS_DEF(AF_INET):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET),
        &reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_addr,
        addr_string, sizeof(addr_string), 0, op->ec_);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_port);
    break;
  case ASIO_OS_DEF(AF_INET6):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET6),
        &reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_addr,
        addr_string, sizeof(addr_string), 0, op->ec_);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_port);
    break;
  default:
    op->ec_ = asio::error::address_family_not_supported;
    break;
  }

  if (op->ec_)
  {
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  try
  {
    async_manager_.async(impl.socket_->ConnectAsync(
          ref new Windows::Networking::HostName(
            winrt_utils::string(addr_string)),
          winrt_utils::string(port)), op);
  }
  catch (Platform::Exception^ e)
  {
    op->ec_ = asio::error_code(
        e->HResult, asio::system_category());
    io_service_.post_immediate_completion(op, is_continuation);
  }
}

asio::error_code winrt_dsocket_service_base::do_bind(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const void* addr, asio::error_code& ec)
{
  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return ec;
  }

  char addr_string[max_addr_v6_str_len];
  bool addr_unspecified = false;
  unsigned short port;
  switch (reinterpret_cast<const socket_addr_type*>(addr)->sa_family)
  {
  case ASIO_OS_DEF(AF_INET):
    if (reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_addr.s_addr
      == 0)
      addr_unspecified = true;
    else
      socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET),
        &reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_addr,
        addr_string, sizeof(addr_string), 0, ec);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_port);
    break;
  case ASIO_OS_DEF(AF_INET6):
    if (
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[0] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[1] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[2] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[3] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[4] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[5] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[6] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[7] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[8] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[9] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[10] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[11] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[12] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[13] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[14] == 0 &&
      reinterpret_cast<const
      sockaddr_in6_type*>(addr)->sin6_addr.s6_addr[15] == 0
    )
      addr_unspecified = true;
    else
      socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET6),
        &reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_addr,
        addr_string, sizeof(addr_string), 0, ec);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_port);
    break;
  default:
    ec = asio::error::address_family_not_supported;
    return ec;
  }

  if (!ec) try
  {
    async_manager_.sync(impl.socket_->BindEndpointAsync(
      addr_unspecified ? nullptr : ref new
      Windows::Networking::HostName(winrt_utils::string(addr_string)),
      port ? winrt_utils::string(port) : ""), ec);
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
  }

  return ec;
}

std::size_t winrt_dsocket_service_base::do_send(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const asio::const_buffer& data,
    socket_base::message_flags flags, asio::error_code& ec)
{
  if (flags)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  try
  {
    buffer_sequence_adapter<asio::const_buffer,
      asio::const_buffers_1> bufs(asio::buffer(data));

    if (bufs.all_empty())
    {
      ec = asio::error_code();
      return 0;
    }

    return async_manager_.sync(
        impl.socket_->OutputStream->WriteAsync(bufs.buffers()[0]), ec);
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
    return 0;
  }
}

void winrt_dsocket_service_base::start_send_op(
      winrt_dsocket_service_base::base_implementation_type& impl,
      const asio::const_buffer& data, socket_base::message_flags flags,
      winrt_async_op<unsigned int>* op, bool is_continuation)
{
  if (flags)
  {
    op->ec_ = asio::error::operation_not_supported;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!is_open(impl))
  {
    op->ec_ = asio::error::bad_descriptor;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  try
  {
    buffer_sequence_adapter<asio::const_buffer,
        asio::const_buffers_1> bufs(asio::buffer(data));

    if (bufs.all_empty())
    {
      io_service_.post_immediate_completion(op, is_continuation);
      return;
    }

    async_manager_.async(
        impl.socket_->OutputStream->WriteAsync(bufs.buffers()[0]), op);
  }
  catch (Platform::Exception^ e)
  {
    op->ec_ = asio::error_code(e->HResult,
        asio::system_category());
    io_service_.post_immediate_completion(op, is_continuation);
  }
}

std::size_t winrt_dsocket_service_base::do_send_to(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const asio::const_buffer& data, const void* addr,
    socket_base::message_flags flags, asio::error_code& ec)
{
  if (flags)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  char addr_string[max_addr_v6_str_len];
  unsigned short port;
  switch (reinterpret_cast<const socket_addr_type*>(addr)->sa_family)
  {
  case ASIO_OS_DEF(AF_INET):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET),
        &reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_addr,
        addr_string, sizeof(addr_string), 0, ec);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_port);
    break;
  case ASIO_OS_DEF(AF_INET6):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET6),
        &reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_addr,
        addr_string, sizeof(addr_string), 0, ec);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_port);
    break;
    default:
        ec = asio::error::address_family_not_supported;
        return 0;
  }

  if (!ec) try
  {
    buffer_sequence_adapter<asio::const_buffer,
        asio::const_buffers_1> bufs(asio::buffer(data));

    if (bufs.all_empty())
    {
      ec = asio::error_code();
      return 0;
    }

    Windows::Storage::Streams::IOutputStream^ output_stream = async_manager_.sync(
        impl.socket_->GetOutputStreamAsync(
            ref new Windows::Networking::HostName(
                winrt_utils::string(addr_string)),
                winrt_utils::string(port)), ec);
    if(!ec)
      return async_manager_.sync(output_stream->WriteAsync(bufs.buffers()[0]), ec);
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
  }

  return 0;
}

void winrt_dsocket_service_base::start_send_to_op(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const asio::const_buffer& data, const void* addr,
    socket_base::message_flags flags,
    winrt_async_op<unsigned int>* op, bool is_continuation)
{
  if (flags)
  {
    op->ec_ = asio::error::operation_not_supported;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!is_open(impl))
  {
    op->ec_ = asio::error::bad_descriptor;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  char addr_string[max_addr_v6_str_len];
  unsigned short port;
  switch (reinterpret_cast<const socket_addr_type*>(addr)->sa_family)
  {
  case ASIO_OS_DEF(AF_INET):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET),
        &reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_addr,
        addr_string, sizeof(addr_string), 0, op->ec_);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in4_type*>(addr)->sin_port);
    break;
  case ASIO_OS_DEF(AF_INET6):
    socket_ops::inet_ntop(ASIO_OS_DEF(AF_INET6),
        &reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_addr,
        addr_string, sizeof(addr_string), 0, op->ec_);
    port = socket_ops::network_to_host_short(
        reinterpret_cast<const sockaddr_in6_type*>(addr)->sin6_port);
    break;
  default:
    op->ec_ = asio::error::address_family_not_supported;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (op->ec_)
  {
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  try
  {
    buffer_sequence_adapter<asio::const_buffer,
        asio::const_buffers_1> bufs(asio::buffer(data));

    if (bufs.all_empty())
    {
      io_service_.post_immediate_completion(op, is_continuation);
      return;
    }

    Windows::Storage::Streams::IOutputStream^ output_stream = async_manager_.sync(
        impl.socket_->GetOutputStreamAsync(
            ref new Windows::Networking::HostName(
                winrt_utils::string(addr_string)),
                winrt_utils::string(port)), op->ec_);

    if (op->ec_)
    {
      io_service_.post_immediate_completion(op, is_continuation);
    }
    else
    {
      async_manager_.async(
        output_stream->WriteAsync(bufs.buffers()[0]), op);
    }
  }
  catch (Platform::Exception^ e)
  {
    op->ec_ = asio::error_code(e->HResult,
        asio::system_category());
    io_service_.post_immediate_completion(op, is_continuation);
  }
}

std::size_t winrt_dsocket_service_base::do_receive(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const asio::mutable_buffer& data,
    socket_base::message_flags flags, asio::error_code& ec)
{
  if (flags)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  std::size_t buffer_size = asio::buffer_size(data);
  if (buffer_size == 0)
  {
    ec = asio::error_code();
    return 0;
  }

  try
  {
    Windows::Networking::Sockets::DatagramSocketMessageReceivedEventArgs^ recv_args;
    {
      asio::detail::mutex::scoped_lock l(impl.receive_handler_->recv_mutex_);
      impl.receive_handler_->recv_event_.wait(l);
      recv_args = impl.receive_handler_->recv_queue_.front();
      impl.receive_handler_->recv_queue_.pop();
      if (impl.receive_handler_->recv_queue_.empty())
        impl.receive_handler_->recv_event_.clear(l);
    }

    Windows::Storage::Streams::DataReader^ data_reader = recv_args->GetDataReader();
    unsigned int bytes_available = data_reader->UnconsumedBufferLength;

    Platform::ArrayReference<unsigned char> buf(asio::buffer_cast<unsigned char *>(data),
        buffer_size);
    data_reader->ReadBytes(buf);

    std::size_t bytes_transferred = bytes_available;
    if (bytes_transferred == 0)
    {
      ec = asio::error::eof;
    }
    else if (buffer_size < bytes_available)
    {
      ec = asio::error::message_size;
      bytes_transferred = buffer_size;
    }

    return bytes_transferred;
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
    return 0;
  }
}

void winrt_dsocket_service_base::start_receive_op(
    winrt_dsocket_service_base::base_implementation_type& impl,
    const asio::mutable_buffer& data, void* addr,
    socket_base::message_flags flags,
    winrt_async_op<Windows::Storage::Streams::IBuffer^>* op,
    bool is_continuation)
{
  if (flags)
  {
    op->ec_ = asio::error::operation_not_supported;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  if (!is_open(impl))
  {
    op->ec_ = asio::error::bad_descriptor;
    io_service_.post_immediate_completion(op, is_continuation);
    return;
  }

  try
  {
    buffer_sequence_adapter<asio::mutable_buffer,
      asio::mutable_buffers_1> bufs(asio::buffer(data));

    if (bufs.all_empty())
    {
      io_service_.post_immediate_completion(op, is_continuation);
      return;
    }

    Windows::Networking::Sockets::DatagramSocketMessageReceivedEventArgs ^recv_args = nullptr;
    {
      asio::detail::mutex::scoped_lock l(impl.receive_handler_->recv_mutex_);
      if (!impl.receive_handler_->recv_queue_.empty())
      {
        recv_args = impl.receive_handler_->recv_queue_.front();
        impl.receive_handler_->recv_queue_.pop();
        if (impl.receive_handler_->recv_queue_.empty())
          impl.receive_handler_->recv_event_.clear(l);
      }
      else
      {
        impl.receive_handler_->recv_callback_ = std::bind(&winrt_dsocket_service_base::finish_receive_op,
            this, bufs.buffers()[0], addr, op, is_continuation, std::placeholders::_1);
      }
    }

    if (recv_args)
      finish_receive_op(bufs.buffers()[0], addr, op, is_continuation, recv_args);
  }
  catch (Platform::Exception^ e)
  {
    op->ec_ = asio::error_code(e->HResult,
        asio::system_category());
    io_service_.post_immediate_completion(op, is_continuation);
  }
}

void winrt_dsocket_service_base::finish_receive_op(
    Windows::Storage::Streams::IBuffer^ data, void* addr,
    winrt_async_op<Windows::Storage::Streams::IBuffer^>* op, bool is_continuation,
    Windows::Networking::Sockets::DatagramSocketMessageReceivedEventArgs^ recv_args)
{
  try
  {
    if (addr != nullptr)
    {
      u_short_type remote_port = socket_ops::host_to_network_short(
        winrt_utils::integer(recv_args->RemotePort));
      Windows::Networking::HostName^ host_name = recv_args->RemoteAddress;
      switch(host_name->Type)
      {
      case Windows::Networking::HostNameType::Ipv4:
        reinterpret_cast<sockaddr_in4_type*>(addr)->sin_family = ASIO_OS_DEF(AF_INET);
        reinterpret_cast<sockaddr_in4_type*>(addr)->sin_port = remote_port;
        socket_ops::inet_pton(ASIO_OS_DEF(AF_INET),
            winrt_utils::string(host_name->RawName).c_str(),
            &reinterpret_cast<sockaddr_in4_type*>(addr)->sin_addr, nullptr, op->ec_);
        break;
      case Windows::Networking::HostNameType::Ipv6:
        reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_family = ASIO_OS_DEF(AF_INET6);
        reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_port = remote_port;
        socket_ops::inet_pton(ASIO_OS_DEF(AF_INET6),
            winrt_utils::string(host_name->RawName).c_str(),
            &reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_addr,
            reinterpret_cast<unsigned long *>(&reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_scope_id),
            op->ec_);
        break;
      default:
        op->ec_ = asio::error::address_family_not_supported;
        io_service_.post_immediate_completion(op, is_continuation);
        return;
      }

      if (op->ec_)
      {
        io_service_.post_immediate_completion(op, is_continuation);
        return;
      }
    }

    if (recv_args->GetDataReader()->UnconsumedBufferLength > data->Capacity)
    {
      op->ec_ = asio::error::message_size;
      io_service_.post_immediate_completion(op, is_continuation);
      return;
    }

    Windows::Storage::Streams::IInputStream^ stream = recv_args->GetDataStream();
    async_manager_.async(
      stream->ReadAsync(data, data->Capacity,
        Windows::Storage::Streams::InputStreamOptions::Partial), op);
  }
  catch (Platform::Exception^ e)
  {
    op->ec_ = asio::error_code(e->HResult,
      asio::system_category());
    io_service_.post_immediate_completion(op, is_continuation);
  }
}

std::size_t winrt_dsocket_service_base::do_receive_from(base_implementation_type& impl,
    const asio::mutable_buffer& data, void* addr,
    socket_base::message_flags flags, asio::error_code& ec)
{
  if (flags)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  if (!is_open(impl))
  {
    ec = asio::error::bad_descriptor;
    return 0;
  }

  std::size_t buffer_size = asio::buffer_size(data);
  if (buffer_size == 0)
  {
    ec = asio::error_code();
    return 0;
  }

  try
  {
    Windows::Networking::Sockets::DatagramSocketMessageReceivedEventArgs^ recv_args;
    {
      asio::detail::mutex::scoped_lock l(impl.receive_handler_->recv_mutex_);
      impl.receive_handler_->recv_event_.wait(l);
      recv_args = impl.receive_handler_->recv_queue_.front();
      impl.receive_handler_->recv_queue_.pop();
      if (impl.receive_handler_->recv_queue_.empty())
        impl.receive_handler_->recv_event_.clear(l);
    }

    u_short_type remote_port = socket_ops::host_to_network_short(
        winrt_utils::integer(recv_args->RemotePort));
    Windows::Networking::HostName^ host_name = recv_args->RemoteAddress;
    switch(host_name->Type)
    {
    case Windows::Networking::HostNameType::Ipv4:
      reinterpret_cast<sockaddr_in4_type*>(addr)->sin_family = ASIO_OS_DEF(AF_INET);
      reinterpret_cast<sockaddr_in4_type*>(addr)->sin_port = remote_port;
      socket_ops::inet_pton(ASIO_OS_DEF(AF_INET),
          winrt_utils::string(host_name->RawName).c_str(),
          &reinterpret_cast<sockaddr_in4_type*>(addr)->sin_addr, nullptr, ec);
      break;
    case Windows::Networking::HostNameType::Ipv6:
      reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_family = ASIO_OS_DEF(AF_INET6);
      reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_port = remote_port;
      socket_ops::inet_pton(ASIO_OS_DEF(AF_INET6),
          winrt_utils::string(host_name->RawName).c_str(),
          &reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_addr,
          reinterpret_cast<unsigned long *>(&reinterpret_cast<sockaddr_in6_type*>(addr)->sin6_scope_id), ec);
      break;
    default:
      ec = asio::error::address_family_not_supported;
      return 0;
    }

    if (ec)
      return 0;

    Windows::Storage::Streams::DataReader^ data_reader = recv_args->GetDataReader();
    unsigned int bytes_available = data_reader->UnconsumedBufferLength;

    Platform::ArrayReference<unsigned char> buf(asio::buffer_cast<unsigned char *>(data),
        buffer_size);
    data_reader->ReadBytes(buf);

    std::size_t bytes_transferred = bytes_available;
    if (bytes_transferred == 0)
    {
      ec = asio::error::eof;
    }
    else if (buffer_size < bytes_available)
    {
      ec = asio::error::message_size;
      bytes_transferred = buffer_size;
    }

    return bytes_transferred;
  }
  catch (Platform::Exception^ e)
  {
    ec = asio::error_code(e->HResult,
        asio::system_category());
    return 0;
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_IMPL_WINRT_DSOCKET_SERVICE_BASE_IPP
