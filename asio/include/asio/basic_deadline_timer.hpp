//
// basic_deadline_timer.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <cstddef>
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/service_factory.hpp"
#include "asio/detail/noncopyable.hpp"

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
 * Async_Object, Error_Source.
 *
 * @sa @ref deadline_timer_reset
 *
 * @par Examples:
 * Performing a blocking wait:
 * @code
 * // Construct a timer without setting an expiry time.
 * asio::deadline_timer timer(io_service);
 *
 * // Set an expiry time relative to now.
 * timer.expires_from_now(boost::posix_time::seconds(5));
 *
 * // Wait for the timer to expire.
 * timer.wait();
 * @endcode
 *
 * @par 
 * Performing an asynchronous wait:
 * @code
 * void handler(const asio::error& error)
 * {
 *   if (!error)
 *   {
 *     // Timer expired.
 *   }
 * }
 *
 * ...
 *
 * // Construct a timer with an absolute expiry time.
 * asio::deadline_timer timer(io_service,
 *     boost::posix_time::time_from_string("2005-12-07 23:59:59.000"));
 *
 * // Start an asynchronous wait.
 * timer.async_wait(handler);
 * @endcode
 */
template <typename Service>
class basic_deadline_timer
  : private noncopyable
{
public:
  /// The type of the service that will be used to provide timer operations.
  typedef Service service_type;

  /// The native implementation type of the timer.
  typedef typename service_type::impl_type impl_type;

  /// The io_service type for this type.
  typedef typename service_type::io_service_type io_service_type;

  /// The type used for reporting errors.
  typedef asio::error error_type;

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
   * @param io_service The io_service object that the timer will use to dispatch
   * handlers for any asynchronous operations performed on the timer.
   */
  explicit basic_deadline_timer(io_service_type& io_service)
    : service_(io_service.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
  }

  /// Constructor to set a particular expiry time as an absolute time.
  /**
   * This constructor creates a timer and sets the expiry time.
   *
   * @param io_service The io_service object that the timer will use to dispatch
   * handlers for any asynchronous operations performed on the timer.
   *
   * @param expiry_time The expiry time to be used for the timer, expressed
   * as an absolute time.
   */
  basic_deadline_timer(io_service_type& io_service,
      const time_type& expiry_time)
    : service_(io_service.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
    destroy_on_block_exit auto_destroy(service_, impl_);
    service_.expires_at(impl_, expiry_time);
    auto_destroy.cancel();
  }

  /// Constructor to set a particular expiry time relative to now.
  /**
   * This constructor creates a timer and sets the expiry time.
   *
   * @param io_service The io_service object that the timer will use to dispatch
   * handlers for any asynchronous operations performed on the timer.
   *
   * @param expiry_time The expiry time to be used for the timer, relative to
   * now.
   */
  basic_deadline_timer(io_service_type& io_service,
      const duration_type& expiry_time)
    : service_(io_service.get_service(service_factory<Service>())),
      impl_(service_.null())
  {
    service_.create(impl_);
    destroy_on_block_exit auto_destroy(service_, impl_);
    service_.expires_from_now(impl_, expiry_time);
    auto_destroy.cancel();
  }

  /// Destructor.
  ~basic_deadline_timer()
  {
    service_.destroy(impl_);
  }

  /// Get the io_service associated with the object.
  /**
   * This function may be used to obtain the io_service object that the timer
   * uses to dispatch handlers for asynchronous operations.
   *
   * @return A reference to the io_service object that the timer will use to
   * dispatch handlers. Ownership is not transferred to the caller.
   */
  io_service_type& io_service()
  {
    return service_.io_service();
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
   * Whether the timer has expired or not does not affect this value.
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
   *
   * @note Modifying the expiry time of a timer while it is active (where
   * active means there are asynchronous waits on the timer) has undefined
   * behaviour. See @ref deadline_timer_reset for information on how to safely
   * alter a timer's expiry in this case.
   */
  void expires_at(const time_type& expiry_time)
  {
    service_.expires_at(impl_, expiry_time);
  }

  /// Get the timer's expiry time relative to now.
  /**
   * This function may be used to obtain the timer's current expiry time.
   * Whether the timer has expired or not does not affect this value.
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
   *
   * @note Modifying the expiry time of a timer while it is active (where
   * active means there are asynchronous waits on the timer) has undefined
   * behaviour. See @ref deadline_timer_reset for information on how to safely
   * alter a timer's expiry in this case.
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
   * Cancelling the timer does not change the expiry time.
   *
   * @return The number of asynchronous operations that were cancelled.
   */
  std::size_t cancel()
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
   * timer. It always returns immediately.
   *
   * For each call to async_wait(), the supplied handler will be called exactly
   * once. The handler will be called when:
   *
   * @li The timer has expired.
   *
   * @li The timer was cancelled, in which case the handler is passed the error
   * code asio::error::operation_aborted.
   *
   * @param handler The handler to be called when the timer expires. Copies
   * will be made of the handler as required. The function signature of the
   * handler must be:
   * @code void handler(
   *   const asio::error& error // Result of operation
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. Invocation
   * of the handler will be performed in a manner equivalent to using
   * asio::io_service::post().
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

  // Helper class to automatically destroy the implementation on block exit.
  class destroy_on_block_exit
  {
  public:
    destroy_on_block_exit(service_type& service, impl_type& impl)
      : service_(&service), impl_(impl)
    {
    }

    ~destroy_on_block_exit()
    {
      if (service_)
      {
        service_->destroy(impl_);
      }
    }

    void cancel()
    {
      service_ = 0;
    }

  private:
    service_type* service_;
    impl_type& impl_;
  };
};

/**
 * @page deadline_timer_reset Changing an active deadline_timer's expiry time
 *
 * Changing the expiry time of a timer while there are asynchronous waits on it
 * has undefined behaviour. To safely change a timer's expiry, pending
 * asynchronous waits need to be cancelled first. This works as follows:
 *
 * @li The asio::basic_deadline_timer::cancel() function returns the
 * number of asynchronous waits that were cancelled. If it returns 0 then you
 * were too late and the wait handler has already been executed, or will soon be
 * executed. If it returns 1 then the wait handler was successfully cancelled.
 *
 * @li If a wait handler is cancelled, the asio::error passed to it
 * contains the value asio::error::operation_aborted.
 *
 * For example, to reset a timer's expiry time in response to some event you
 * would do something like this:
 *
 * @code
 * void on_some_event()
 * {
 *   if (my_timer.cancel() > 0)
 *   {
 *     // We managed to cancel the timer. Set new expiry time.
 *     my_timer.expires_from_now(seconds(5));
 *     my_timer.async_wait(on_timeout);
 *   }
 *   else
 *   {
 *     // Too late, timer has already expired!
 *   }
 * }
 *
 * void on_timeout(const asio::error& e)
 * {
 *   if (e != asio::error::operation_aborted)
 *   {
 *     // Timer was not cancelled, take necessary action.
 *   }
 * }
 * @endcode
 *
 * @sa asio::basic_deadline_timer
 */

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_BASIC_DEADLINE_TIMER_HPP
