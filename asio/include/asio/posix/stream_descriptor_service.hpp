//
// stream_descriptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_POSIX_STREAM_DESCRIPTOR_SERVICE_HPP
#define ASIO_POSIX_STREAM_DESCRIPTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/epoll_reactor.hpp"
#include "asio/detail/kqueue_reactor.hpp"
#include "asio/detail/select_reactor.hpp"
#include "asio/detail/service_base.hpp"
#include "asio/detail/reactive_descriptor_service.hpp"

#if !defined(ASIO_DISABLE_POSIX_STREAM_DESCRIPTOR)
# if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#  define ASIO_HAS_POSIX_STREAM_DESCRIPTOR 1
# endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
#endif // !defined(ASIO_DISABLE_POSIX_STREAM_DESCRIPTOR)

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
  || defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace posix {

/// Default service implementation for a stream descriptor.
class stream_descriptor_service
#if defined(GENERATING_DOCUMENTATION)
  : public asio::io_service::service
#else
  : public asio::detail::service_base<stream_descriptor_service>
#endif
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// The unique service identifier.
  static asio::io_service::id id;
#endif

private:
  // The type of the platform-specific implementation.
#if defined(ASIO_HAS_EPOLL)
  typedef detail::reactive_descriptor_service<
      detail::epoll_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_KQUEUE)
  typedef detail::reactive_descriptor_service<
      detail::kqueue_reactor<false> > service_impl_type;
#elif defined(ASIO_HAS_DEV_POLL)
  typedef detail::reactive_descriptor_service<
      detail::dev_poll_reactor<false> > service_impl_type;
#else
  typedef detail::reactive_descriptor_service<
      detail::select_reactor<false> > service_impl_type;
#endif

public:
  /// The type of a stream descriptor implementation.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined implementation_type;
#else
  typedef service_impl_type::implementation_type implementation_type;
#endif

  /// The native descriptor type.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_type;
#else
  typedef service_impl_type::native_type native_type;
#endif

  /// Construct a new stream descriptor service for the specified io_service.
  explicit stream_descriptor_service(asio::io_service& io_service)
    : asio::detail::service_base<stream_descriptor_service>(io_service),
      service_impl_(asio::use_service<service_impl_type>(io_service))
  {
  }

  /// Destroy all user-defined descriptorr objects owned by the service.
  void shutdown_service()
  {
  }

  /// Construct a new stream descriptor implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a stream descriptor implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Assign an existing native descriptor to a stream descriptor.
  asio::error_code assign(implementation_type& impl,
      const native_type& native_descriptor, asio::error_code& ec)
  {
    return service_impl_.assign(impl, native_descriptor, ec);
  }

  /// Determine whether the descriptor is open.
  bool is_open(const implementation_type& impl) const
  {
    return service_impl_.is_open(impl);
  }

  /// Close a stream descriptor implementation.
  asio::error_code close(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.close(impl, ec);
  }

  /// Get the native descriptor implementation.
  native_type native(implementation_type& impl)
  {
    return service_impl_.native(impl);
  }

  /// Cancel all asynchronous operations associated with the descriptor.
  asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    return service_impl_.cancel(impl, ec);
  }

  /// Perform an IO control command on the descriptor.
  template <typename IoControlCommand>
  asio::error_code io_control(implementation_type& impl,
      IoControlCommand& command, asio::error_code& ec)
  {
    return service_impl_.io_control(impl, command, ec);
  }

  /// Write the given data to the stream.
  template <typename ConstBufferSequence>
  std::size_t write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    return service_impl_.write_some(impl, buffers, ec);
  }

  /// Start an asynchronous write.
  template <typename ConstBufferSequence, typename WriteHandler>
  void async_write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, WriteHandler descriptorr)
  {
    service_impl_.async_write_some(impl, buffers, descriptorr);
  }

  /// Read some data from the stream.
  template <typename MutableBufferSequence>
  std::size_t read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    return service_impl_.read_some(impl, buffers, ec);
  }

  /// Start an asynchronous read.
  template <typename MutableBufferSequence, typename ReadHandler>
  void async_read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, ReadHandler descriptorr)
  {
    service_impl_.async_read_some(impl, buffers, descriptorr);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace posix
} // namespace asio

#endif // defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR)
       //   || defined(GENERATING_DOCUMENTATION)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_POSIX_STREAM_DESCRIPTOR_SERVICE_HPP
