//
// completion_context_locker.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_COMPLETION_CONTEXT_LOCKER_HPP
#define ASIO_COMPLETION_CONTEXT_LOCKER_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

class completion_context;

/// The completion_context_locker class is a base class intended for any class
/// that needs to acquire and release completion contexts.
class completion_context_locker
{
public:
  /// Destructor.
  virtual ~completion_context_locker();

protected:
  /// Attempt to acquire a completion context. Returns true if the context was
  /// successfully acquired.
  bool try_acquire(completion_context& context);

  /// Wait to acquire a completion context, with a call being made to the
  /// completion_context_acquired() function when it has been successfully
  /// acquired.
  void acquire(completion_context& context, void* arg);

  /// Release a completion context.
  void release(completion_context& context);

private:
  /// Only the completion_context class is permitted to notify a locker object
  /// that it has been successfully acquired.
  friend class completion_context;

  /// Callback function when the context has been acquired.
  virtual void completion_context_acquired(void* arg) throw () = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_COMPLETION_CONTEXT_LOCKER_HPP
