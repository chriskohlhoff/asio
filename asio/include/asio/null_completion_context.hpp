//
// null_completion_context.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_NULL_COMPLETION_CONTEXT_HPP
#define ASIO_NULL_COMPLETION_CONTEXT_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

namespace detail
{
  // Helper template used to create a global instance of the
  // null_completion_context without needing a .cpp file for it.
  template <typename Type>
  struct global
  {
    static Type instance;
  };

  template <typename Type> Type global<Type>::instance;
}

/// The completion_context class is the abstract base class for all completion
/// context implementations. A completion context is used to determine when
/// an upcall can be made to the completion handler of an asynchronous
/// operation.
class null_completion_context
  : private boost::noncopyable
{
public:
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
  static null_completion_context& instance()
  {
    return detail::global<null_completion_context>::instance;
  }

  /// Attempt to acquire the right to make an upcall.
  /**
   * This function is called to attempt to obtain the right to make an upcall
   * to a completion handler. This function always returns a result
   * immediately.
   *
   * If the right to make an upcall was successfully acquired, then a later
   * call must be made to the release() function to relinquish that right.
   *
   * @return Returns true if the right to make an upcall was granted.
   */
  bool try_acquire()
  {
    return true;
  }

  /// Acquire the right to make an upcall.
  /**
   * This function is called to obtain the right to make an upcall to a
   * completion handler. The handler will be called when the right is granted.
   *
   * @param handler The function object to be called when the right is granted.
   */
  template <typename Handler>
  void acquire(Handler handler)
  {
    handler();
  }

  /// Relinquish a previously granted right to make an upcall.
  void release()
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_NULL_COMPLETION_CONTEXT_HPP
