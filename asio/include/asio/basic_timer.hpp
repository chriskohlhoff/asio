//
// basic_timer.hpp
// ~~~~~~~~~~~~~~~
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

#ifndef ASIO_BASIC_TIMER_HPP
#define ASIO_BASIC_TIMER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/null_completion_context.hpp"
#include "asio/service_factory.hpp"
#include "asio/timer_base.hpp"

namespace asio {

/// The basic_timer class template provides asynchronous timer functionality.
/// Most applications will simply use the timer typedef.
template <typename Service>
class basic_timer
  : public timer_base,
    private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide timer operations.
  typedef Service service_type;

  /// The native implementation type of the timer.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// Constructor.
  explicit basic_timer(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
  }

  /// Construct and set to a particular time.
  basic_timer(demuxer_type& d, from_type from_when, long seconds,
      int microseconds = 0)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
    service_.set(impl_, from_when, seconds, microseconds);
  }

  /// Destructor.
  ~basic_timer()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Set the timer.
  void set(from_type from_when, long seconds, long microseconds = 0)
  {
    service_.set(impl_, from_when, seconds, microseconds);
  }

  /// Expire the timer immediately.
  void expire()
  {
    service_.expire(impl_);
  }

  /// Perform a blocking wait on the timer.
  void wait()
  {
    service_.wait(impl_);
  }

  /// Start an asynchronous wait on the timer.
  template <typename Handler>
  void async_wait(Handler handler)
  {
    service_.async_wait(impl_, handler, null_completion_context::instance());
  }

  /// Start an asynchronous wait on the timer.
  template <typename Handler, typename Completion_Context>
  void async_wait(Handler handler, Completion_Context& context)
  {
    service_.async_wait(impl_, handler, context);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_TIMER_HPP
