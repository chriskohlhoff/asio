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

namespace asio {

/// The null_completion_context class is a concrete implementation of the
/// Completion_Context concept. It does not place any limits on the number of
/// concurrent upcalls to completion handlers that may be associated with the
/// context. All instances of this class are equivalent.
class null_completion_context
{
public:
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
