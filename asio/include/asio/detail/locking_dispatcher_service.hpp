//
// locking_dispatcher_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_LOCKING_DISPATCHER_SERVICE_HPP
#define ASIO_DETAIL_LOCKING_DISPATCHER_SERVICE_HPP

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
