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

#include "asio/completion_context_locker.hpp"
#include "asio/completion_context.hpp"

namespace asio {

completion_context_locker::
~completion_context_locker()
{
}

bool
completion_context_locker::
try_acquire(
    completion_context& context)
{
  return context.try_acquire();
}

void
completion_context_locker::
acquire(
    completion_context& context,
    void* arg)
{
  context.acquire(*this, arg);
}

void
completion_context_locker::
release(
    completion_context& context)
{
  context.release();
}

} // namespace asio
