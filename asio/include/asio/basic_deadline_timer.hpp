//
// basic_deadline_timer.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_BASIC_DEADLINE_TIMER_HPP
#define ASIO_BASIC_DEADLINE_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service_factory.hpp"

namespace asio {

/// Provides waitable timer functionality.
/**
 * The basic_deadline_timer class template provides the ability to perform a
 * blocking or asynchronous wait for a timer to expire.
 *
 * Most applications will use the asio::deadline_timer typedef.
 *
 * @par Thread Safety:
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * Async_Object.
 */
template <typename Service>
class basic_deadline_timer
  : private boost::noncopyable
{
public:
  /// The type of the service that will be used to provide timer operations.
  typedef Service service_type;

  /// The native implementation type of the timer.
  typedef typename service_type::impl_type impl_type;

  /// The demuxer type for this asynchronous type.
  typedef typename service_type::demuxer_type demuxer_type;

  /// The time type.
  typedef typename service_type::time_type time_type;

  /// The duration type.
  typedef typename service_type::duration_type duration_type;

  /// Constructor.
  /**
   * This constructor creates a timer without setting an expiry time. The
   * expires_at() or expires_from_now() functions must be called to set an
   * expiry time before the timer can be waited on.
   *
   * @param d The demuxer object that the timer will use to dispatch handlers
   * for any asynchronous operations performed on the timer.
   */
  explicit basic_deadline_timer(demuxer_type& d)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
  }

  /// Constructor to set a particular expiry time as an absolute time.
  /**
   * This constructor creates a timer and sets the expiry time.
   *
   * @param d The demuxer object that the timer will use to dispatch handlers
   * for any asynchronous operations performed on the timer.
   *
   * @param expiry_time The expiry time to be used for the timer, expressed
   * relative to the epoch.
   */
  basic_deadline_timer(demuxer_type& d, const time_type& expiry_time)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
    service_.expires_at(impl_, expiry_time);
  }

  /// Constructor to set a particular expiry time relative to now.
  /**
   * This constructor creates a timer and sets the expiry time.
   *
   * @param d The demuxer object that the timer will use to dispatch handlers
   * for any asynchronous operations performed on the timer.
   *
   * @param expiry_time The expiry time to be used for the timer, expressed
   * relative to the epoch.
   */
  basic_deadline_timer(demuxer_type& d, const duration_type& expiry_time)
    : service_(d.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
    service_.expires_from_now(impl_, expiry_time);
  }

  /// Destructor.
  ~basic_deadline_timer()
  {
    service_.destroy(impl_);
  }

  /// Get the demuxer associated with the asynchronous object.
  /**
   * This function may be used to obtain the demuxer object that the timer uses
   * to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the demuxer object that the timer will use to
   * dispatch handlers. Ownership is not transferred to the caller.
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

  /// Get the timer's expiry time as an absolute time.
  /**
   * This function may be used to obtain the timer's current expiry time.
   */
  time_type expires_at() const
  {
    return service_.expires_at(impl_);
  }

  /// Set the timer's expiry time as an absolute time.
  /**
   * This function sets the expiry time.
   *
   * @param expiry_time The expiry time to be used for the timer.
   */
  void expires_at(const time_type& expiry_time)
  {
    service_.expires_at(impl_, expiry_time);
  }

  /// Get the timer's expiry time relative to now.
  /**
   * This function may be used to obtain the timer's current expiry time.
   */
  duration_type expires_from_now() const
  {
    return service_.expires_from_now(impl_);
  }

  /// Set the timer's expiry time relative to now.
  /**
   * This function sets the expiry time.
   *
   * @param expiry_time The expiry time to be used for the timer.
   */
  void expires_from_now(const duration_type& expiry_time)
  {
    service_.expires_from_now(impl_, expiry_time);
  }

  /// Cancel any asynchronous operations that are waiting on the timer.
  /**
   * This function forces the completion of any pending asynchronous wait
   * operations against the timer. The handler for each cancelled operation
   * will be invoked with the asio::error::operation_aborted error code.
   *
   * @return The number of asynchronous operations that were cancelled.
   */
  int cancel()
  {
    return service_.cancel(impl_);
  }

  /// Perform a blocking wait on the timer.
  /**
   * This function is used to wait for the timer to expire. This function
   * blocks and does not return until the timer has expired.
   *
   * @throws asio::error Thrown on failure.
   */
  void wait()
  {
    service_.wait(impl_);
  }

  /// Start an asynchronous wait on the timer.
  /**
   * This function may be used to initiate an asynchronous wait against the
   * timer. It always returns immediately, but the specified handler will be
   * notified when the timer expires, or if the operation is cancelled.
   *
   * @param handler The handler to be called when the timer expires. Copies
   * will be made of the handler as required. The equivalent function signature
   * of the handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   */
  template <typename Handler>
  void async_wait(Handler handler)
  {
    service_.async_wait(impl_, handler);
  }

private:
  /// The backend service implementation.
  service_type& service_;

  /// The underlying native implementation.
  impl_type impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DEADLINE_TIMER_HPP
