//
// locking_dispatcher_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_LOCKING_DISPATCHER_SERVICE_HPP
#define ASIO_DETAIL_LOCKING_DISPATCHER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/locking_dispatcher_impl.hpp"

namespace asio {
namespace detail {

template <typename Demuxer>
class locking_dispatcher_service
{
public:
  // The native type of the locking dispatcher.
  typedef locking_dispatcher_impl<Demuxer>* impl_type;

  // Return a null locking dispatcher implementation.
  static impl_type null()
  {
    return 0;
  }

  // Constructor.
  locking_dispatcher_service(Demuxer& d)
    : demuxer_(d)
  {
  }

  // The demuxer type for this service.
  typedef Demuxer demuxer_type;

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Create a new locking dispatcher implementation.
  void create(impl_type& impl)
  {
    impl = new locking_dispatcher_impl<Demuxer>;
  }

  // Destroy a locking dispatcher implementation.
  void destroy(impl_type& impl)
  {
    if (impl != null())
    {
      delete impl;
      impl = null();
    }
  }

  // Request a dispatcher to invoke the given handler.
  template <typename Handler>
  void dispatch(impl_type& impl, Handler handler)
  {
    impl->dispatch(demuxer_, handler);
  }

  // Request a dispatcher to invoke the given handler and return immediately.
  template <typename Handler>
  void post(impl_type& impl, Handler handler)
  {
    impl->post(demuxer_, handler);
  }

private:
  // The demuxer used for dispatching handlers.
  Demuxer& demuxer_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_LOCKING_DISPATCHER_SERVICE_HPP
