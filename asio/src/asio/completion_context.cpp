//
// completion_context.cpp
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

#include "asio/completion_context.hpp"
#include "asio/completion_context_locker.hpp"

namespace asio {

completion_context::
~completion_context()
{
}

namespace {

class null_completion_context
  : public completion_context
{
private:
  virtual bool try_acquire();
  virtual void acquire(completion_context_locker& locker, void* arg);
  virtual void release();
};

bool
null_completion_context::
try_acquire()
{
  return true;
}

void
null_completion_context::
acquire(
    completion_context_locker& locker,
    void* arg)
{
  notify_locker(locker, arg);
}

void
null_completion_context::
release()
{
}

null_completion_context the_null_completion_context;

} // namespace

completion_context&
completion_context::
null()
{
  return the_null_completion_context;
}

void
completion_context::
notify_locker(
    completion_context_locker& locker,
    void* arg)
  throw ()
{
  locker.completion_context_acquired(arg);
}

} // namespace asio
