//
// completion_context.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_COMPLETION_CONTEXT_HPP
#define ASIO_COMPLETION_CONTEXT_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

class completion_context_locker;

/// The completion_context class is the abstract base class for all completion
/// context implementations. A completion context is used to determine when
/// an upcall can be made to the completion handler of an asynchronous
/// operations.
class completion_context
  : private boost::noncopyable
{
public:
  /// Destructor.
  virtual ~completion_context();

  /// Obtain the null completion context.
  /**
   * This function can be used to obtain a null completion context
   * implementation. This null implementation provides no control over when
   * completion handler upcalls are made.
   *
   * @return A reference to a null completion context object. The ownership of
   * the object is not transferred to the caller, and the object is guaranteed
   * to be valid for the lifetime of the program.
   */
  static completion_context& null();

protected:
  /// Notify a locker that it has acquired the completion context.
  /**
   * This function is used by completion context implementations to notify a
   * completion_context_locker that it has acquired the right to make an
   * upcall. Calls completion_context_locker::completion_context_acquired().
   *
   * @param locker A reference to the completion_context_locker to be notified.
   *
   * @param arg The argument passed by the completion_context_locker object to
   * the acquire function.
   */
  void notify_locker(completion_context_locker& locker, void* arg) throw ();

private:
  /// Only instances of the completion_context_locker class are permitted to
  /// acquire and release a completion context.
  friend class completion_context_locker;

  /// Attempt to acquire the right to make an upcall.
  /**
   * This function is called by a completion_context_locker object to attempt
   * to obtain the right to make an upcall to a completion handler. This
   * function always returns a result immediately.
   *
   * If the right to make an upcall was successfully acquired, then the a later
   * call must be made to the release() function to relinquish that right.
   *
   * @return Returns true if the right to make an upcall was granted.
   */
  virtual bool try_acquire() = 0;

  /// Acquire the right to make an upcall.
  /**
   * This function is called by a completion_context_locker object to obtain
   * the right to make an upcall to a completion handler. The
   * completion_context_locker::completion_context_acquired() function will be
   * called when the right is granted.
   *
   * @param locker The locker object to be notified when the right is granted.
   * The locker object's lifetime must be guaranteed to continue until the
   * notification is received.
   *
   * @param arg A caller-defined argument to be passed back to the locker when
   * the notification is delivered.
   */
  virtual void acquire(completion_context_locker& locker, void* arg) = 0;

  /// Relinquish a previously granted right to make an upcall.
  /**
   * This function must be called by a completion_context_locker object when it
   * has completed making an upcall to a completion_handler.
   */
  virtual void release() = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_COMPLETION_CONTEXT_HPP
