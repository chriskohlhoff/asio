//
// demuxer_task_thread.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/demuxer_task_thread.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

demuxer_task_thread::
demuxer_task_thread(
    demuxer_task& task,
    void* arg)
  : mutex_(),
    task_(&task),
    arg_(arg),
    stop_(false),
    thread_(boost::bind(&demuxer_task_thread::run_task, this))
{
}

demuxer_task_thread::
~demuxer_task_thread()
{
  boost::mutex::scoped_lock lock(mutex_);
  stop_ = true;
  if (task_)
    task_->interrupt_task(arg_);
  lock.unlock();

  thread_.join();
}

void
demuxer_task_thread::
run_task()
{
  boost::mutex::scoped_lock lock(mutex_);
  while (!stop_)
  {
    task_->prepare_task(arg_);
    lock.unlock();
    boost::xtime xt;
    xt.sec = 1;
    xt.nsec = 0;
    bool finished = task_->execute_task(xt, arg_);
    lock.lock();
    if (finished)
    {
      task_ = 0;
      stop_ = true;
    }
  }
}

} // namespace detail
} // namespace asio
