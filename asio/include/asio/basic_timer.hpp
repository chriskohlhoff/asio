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
/// Most applications will use the timer typedef.
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
  /**
   * This constructor creates a timer without setting an expiry time. The set()
   * function must be called before the timer can be waited on.
   *
   * @param d The demuxer object that the timer will use to deliver completions
   * for any asynchronous operations performed on the timer.
   */
  explicit basic_timer(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
  }

  /// Constructor to set a particular expiry time.
  /**
   * This constructor creates a timer and sets the expiry time.
   *
   * @param d The demuxer object that the timer will use to deliver completions
   * for any asynchronous operations performed on the timer.
   *
   * @param from_when The origin time against which the seconds and
   * microseconds values are measured.
   *
   * @param seconds The number of seconds after the from_when origin that the
   * time should expire.
   *
   * @param microseconds The number of microseconds which, in addition to the
   * seconds value, is used to calculate the expiry time relative to the
   * from_when origin value.
   */
  basic_timer(demuxer_type& d, from_type from_when, long seconds,
      long microseconds = 0)
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
  /**
   * This function may be used to obtain the demuxer object that the timer uses
   * to deliver completions for asynchronous operations.
   *
   * @return A reference to the demuxer object that the timer will use to
   * deliver completion notifications. Ownership is not transferred to the
   * caller.
   */
  demuxer_type& demuxer()
  {
    return service_.demuxer();
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * timer. This is intended to allow access to native timer functionality that
   * is not otherwise provided.
   */
  impl_type impl()
  {
    return impl_;
  }

  /// Set the timer.
  /**
   * This function sets the expiry time.
   *
   * @param from_when The origin time against which the seconds and
   * microseconds values are measured.
   *
   * @param seconds The number of seconds after the from_when origin that the
   * time should expire.
   *
   * @param microseconds The number of microseconds which, in addition to the
   * seconds value, is used to calculate the expiry time relative to the
   * from_when origin value.
   */
  void set(from_type from_when, long seconds, long microseconds = 0)
  {
    service_.set(impl_, from_when, seconds, microseconds);
  }

  /// Expire the timer immediately.
  /**
   * This function causes the timer to expire immediately. If there is a
   * pending asynchronous wait operation against the timer it will be forced to
   * complete.
   */
  void expire()
  {
    service_.expire(impl_);
  }

  /// Perform a blocking wait on the timer.
  /**
   * This function is used to wait for the timer to expire. This function
   * blocks and does not return until the timer has expired.
   */
  void wait()
  {
    service_.wait(impl_);
  }

  /// Start an asynchronous wait on the timer.
  /**
   * This function may be used to initiate an asynchronous wait against the
   * timer. It always returns immediately, but the specified handler will be
   * notified when the timer expires.
   *
   * @param handler The completion handler to be called when the timer expires.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   */
  template <typename Handler>
  void async_wait(Handler handler)
  {
    service_.async_wait(impl_, handler, null_completion_context::instance());
  }

  /// Start an asynchronous wait on the timer.
  /**
   * This function may be used to initiate an asynchronous wait against the
   * timer. It always returns immediately, but the specified handler will be
   * notified when the timer expires.
   *
   * @param handler The completion handler to be called when the timer expires.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be: @code void handler(); @endcode
   *
   * @param context The completion context which controls the number of
   * concurrent invocations of handlers that may be made. Ownership of the
   * object is retained by the caller, which must guarantee that it is valid
   * until after the handler has been called.
   */
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
