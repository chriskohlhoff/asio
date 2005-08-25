//
// locking_dispatcher_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LOCKING_DISPATCHER_SERVICE_HPP
#define ASIO_LOCKING_DISPATCHER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/detail/locking_dispatcher_service.hpp"

namespace asio {

/// Default service implementation for a locking dispatcher.
template <typename Allocator = std::allocator<void> >
class locking_dispatcher_service
  : private boost::noncopyable
{
public:
  /// The demuxer type for this service.
  typedef basic_demuxer<demuxer_service<Allocator> > demuxer_type;

private:
  // The type of the platform-specific implementation.
  typedef detail::locking_dispatcher_service<demuxer_type> service_impl_type;

public:
  /// The type of the locking dispatcher.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined impl_type;
#else
  typedef typename service_impl_type::impl_type impl_type;
#endif

  /// Constructor.
  locking_dispatcher_service(demuxer_type& demuxer)
    : service_impl_(demuxer.get_service(service_factory<service_impl_type>()))
  {
  }

  /// Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return service_impl_.demuxer();
  }

  /// Return a null locking dispatcher implementation.
  impl_type null() const
  {
    return service_impl_.null();
  }

  /// Create a new locking dispatcher implementation.
  void create(impl_type& impl)
  {
    service_impl_.create(impl);
  }

  /// Destroy a locking dispatcher implementation.
  void destroy(impl_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Request a dispatcher to invoke the given handler.
  template <typename Handler>
  void dispatch(impl_type& impl, Handler handler)
  {
    service_impl_.dispatch(impl, handler);
  }

  /// Request a dispatcher to invoke the given handler and return immediately.
  template <typename Handler>
  void post(impl_type& impl, Handler handler)
  {
    service_impl_.post(impl, handler);
  }

private:
  // The service that provides the platform-specific implementation.
  service_impl_type& service_impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_LOCKING_DISPATCHER_SERVICE_HPP
