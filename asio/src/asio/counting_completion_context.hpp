//
// counting_completion_context.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_COUNTING_COMPLETION_CONTEXT_HPP
#define ASIO_COUNTING_COMPLETION_CONTEXT_HPP

#include "asio/completion_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// The counting_completion_context class is a concrete implementation of the
/// completion_context class that allows a limitation on the number of
/// concurrent upcalls to completion handlers that may be associated with the
/// context.
class counting_completion_context
  : public completion_context
{
public:
  /// Constructor.
  explicit counting_completion_context(int max_concurrent_upcalls);

  /// Destructor.
  virtual ~counting_completion_context();

private:
  /// Attempt to acquire the right to make an upcall.
  virtual bool try_acquire();

  /// Acquire the right to make an upcall.
  virtual void acquire(completion_context_locker& locker, void* arg);

  /// Relinquish a previously granted right to make an upcall.
  virtual void release();

  // The underlying implementation.
  struct impl;
  impl* impl_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_COUNTING_COMPLETION_CONTEXT_HPP
