//
// random_access_handle_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WINDOWS_RANDOM_ACCESS_HANDLE_SERVICE_HPP
#define ASIO_WINDOWS_RANDOM_ACCESS_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/win_iocp_handle_service.hpp"

#if !defined(ASIO_DISABLE_WINDOWS_RANDOM_ACCESS_HANDLE)
# if defined(ASIO_HAS_IOCP)
#  define ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE 1
# endif // defined(ASIO_HAS_IOCP)
#endif // !defined(ASIO_DISABLE_WINDOWS_RANDOM_ACCESS_HANDLE)

#if defined(ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE) \
  || defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace windows {

/// Default service implementation for a random-access handle.
class random_access_handle_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_service::service
#else
  : public asio::detail::service_base<random_access_handle_service>
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_service::id id;
#endif

private:
  // The type of the platform-specific implementation.
  typedef detail::win_iocp_handle_service service_impl_type;

public:
  /// The type of a random-access handle implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef service_impl_type::implementation_type implementation_type;
#endif

  /// The native handle type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_type;
#else
  typedef service_impl_type::native_type native_type;
#endif

  /// Construct a new random-access handle service for the specified io_service.
  explicit random_access_handle_service(asio::io_service& io_service)
    : asio::detail::service_base<
        random_access_handle_service>(io_service),
      service_impl_(asio::use_service<service_impl_type>(io_service))
  {
  }

  /// Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  /// Construct a new random-access handle implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a random-access handle implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Assign an existing native handle to a random-access handle.
  asio::error_code assign(implementation_type& impl,
      const native_type& native_handle, asio::error_code& ec)
  {
    return service_impl_.assign(impl, native_handle, ec);
  }

  /// Determine whether the handle is open.
  bool is_open(const implementation_type& impl) const
  {
    return service_impl_.is_open(impl);
  }

  /// Close a random-access handle implementation.
  asio::error_code close(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.close(impl, ec);
  }

  /// Get the native handle implementation.
  native_type native(implementation_type& impl)
  {
    return service_impl_.native(impl);
  }

  /// Cancel all asynchronous operations associated with the handle.
  asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.cancel(impl, ec);
  }

  /// Write the given data at the specified offset.
  template <typename ConstBufferSequence>
  std::size_t write_some_at(implementation_type& impl, boost::uint64_t offset,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    return service_impl_.write_some_at(impl, offset, buffers, ec);
  }

  /// Start an asynchronous write at the specified offset.
  template <typename ConstBufferSequence, typename WriteHandler>
  void async_write_some_at(implementation_type& impl, boost::uint64_t offset,
      const ConstBufferSequence& buffers, WriteHandler handler)
  {
    service_impl_.async_write_some_at(impl, offset, buffers, handler);
  }

  /// Read some data from the specified offset.
  template <typename MutableBufferSequence>
  std::size_t read_some_at(implementation_type& impl, boost::uint64_t offset,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    return service_impl_.read_some_at(impl, offset, buffers, ec);
  }

  /// Start an asynchronous read at the specified offset.
  template <typename MutableBufferSequence, typename ReadHandler>
  void async_read_some_at(implementation_type& impl, boost::uint64_t offset,
      const MutableBufferSequence& buffers, ReadHandler handler)
  {
    service_impl_.async_read_some_at(impl, offset, buffers, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace windows
} // namespace asio

#endif // defined(ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE)
       //   || defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_WINDOWS_RANDOM_ACCESS_HANDLE_SERVICE_HPP
