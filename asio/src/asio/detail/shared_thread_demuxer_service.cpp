//
// shared_thread_demuxer_service.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/shared_thread_demuxer_service.hpp"

#include "asio/detail/push_options.hpp"
#include <cassert>
#include <memory>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

shared_thread_demuxer_service::
shared_thread_demuxer_service(
    basic_demuxer<shared_thread_demuxer_service>&)
  : mutex_(),
    waiting_tasks_(),
    running_tasks_(),
    outstanding_operations_(0),
    ready_completions_(),
    interrupted_(false),
    current_thread_in_pool_(),
    idle_thread_count_(0),
    idle_thread_condition_()
{
}

shared_thread_demuxer_service::
~shared_thread_demuxer_service()
{
}

void
shared_thread_demuxer_service::
run()
{
  current_thread_in_pool_ = true;

  boost::mutex::scoped_lock lock(mutex_);

  while (!interrupted_ && outstanding_operations_ > 0)
  {
    if (!ready_completions_.empty())
    {
      completion_info& front_info = ready_completions_.front();
      completion_info info;
      info.handler.swap(front_info.handler);
      info.context = front_info.context;
      ready_completions_.pop();
      lock.unlock();
      do_completion_upcall(info.handler);
      release(*info.context);
      lock.lock();
      assert(outstanding_operations_ > 0);
      --outstanding_operations_;
    }
    else if (!waiting_tasks_.empty())
    {
      // Move the task to the running state.
      task_list::iterator current_task = waiting_tasks_.begin();
      running_tasks_.splice(running_tasks_.end(), waiting_tasks_,
          current_task);

      // Run the task.
      current_task->task->prepare_task(current_task->arg);
      lock.unlock();
      boost::xtime interval;
      interval.sec = 1;
      interval.nsec = 0;
      bool finished = current_task->task->execute_task(interval,
          current_task->arg);
      lock.lock();

      // Execution finished, so remove it from the list of running tasks.
      if (finished)
      {
        running_tasks_.erase(current_task);
      }
      else
      {
        waiting_tasks_.splice(waiting_tasks_.end(), running_tasks_,
            current_task);
      }
    }
    else 
    {
      // No tasks to run right now, so just wait for work to do.
      ++idle_thread_count_;
      idle_thread_condition_.wait(lock);
      --idle_thread_count_;
    }
  }

  if (!interrupted_)
  {
    // No more work to do!
    interrupt_all_threads();
  }

  current_thread_in_pool_ = false;
}

void
shared_thread_demuxer_service::
interrupt()
{
  boost::mutex::scoped_lock lock(mutex_);

  interrupt_all_threads();
}

void
shared_thread_demuxer_service::
reset()
{
  boost::mutex::scoped_lock lock(mutex_);

  interrupted_ = false;
}

void
shared_thread_demuxer_service::
add_task(
    demuxer_task& task,
    void* arg)
{
  boost::mutex::scoped_lock lock(mutex_);

  task_info info = { &task, arg };
  waiting_tasks_.push_back(info);

  interrupt_one_idle_thread();
}

void
shared_thread_demuxer_service::
operation_started()
{
  boost::mutex::scoped_lock lock(mutex_);

  ++outstanding_operations_;
}

void
shared_thread_demuxer_service::
operation_completed(
    const completion_handler& handler,
    completion_context& context,
    bool allow_nested_delivery)
{
  boost::mutex::scoped_lock lock(mutex_);

  if (try_acquire(context))
  {
    if (allow_nested_delivery && current_thread_in_pool_)
    {
      lock.unlock();
      do_completion_upcall(handler);
      release(context);
      lock.lock();
      if (--outstanding_operations_ == 0)
        interrupt_all_threads();
    }
    else
    {
      completion_info info;
      info.handler = handler;
      info.context = &context;
      ready_completions_.push(info);
      if (!interrupt_one_idle_thread())
        interrupt_one_task();
    }
  }
  else
  {
    completion_info* info = new completion_info;
    info->handler = handler;
    info->context = &context;
    acquire(context, info);
  }
}

void
shared_thread_demuxer_service::
operation_immediate(
    const completion_handler& handler,
    completion_context& context,
    bool allow_nested_delivery)
{
  operation_started();
  operation_completed(handler, context, allow_nested_delivery);
}

void
shared_thread_demuxer_service::
completion_context_acquired(
    void* arg)
  throw ()
{
  try
  {
    boost::mutex::scoped_lock lock(mutex_);

    std::auto_ptr<completion_info> info(static_cast<completion_info*>(arg));
    ready_completions_.push(*info);

    if (!interrupt_one_idle_thread())
      interrupt_one_task();
  }
  catch (...)
  {
  }
}

void
shared_thread_demuxer_service::
interrupt_all_threads()
{
  interrupted_ = true;

  idle_thread_condition_.notify_all();

  task_list::iterator i = running_tasks_.begin();
  task_list::iterator end = running_tasks_.end();
  while (i != end)
  {
    i->task->interrupt_task(i->arg);
    ++i;
  }
}

bool
shared_thread_demuxer_service::
interrupt_one_idle_thread()
{
  if (idle_thread_count_ > 0)
  {
    idle_thread_condition_.notify_one();
    return true;
  }

  return false;
}

bool
shared_thread_demuxer_service::
interrupt_one_task()
{
  if (!running_tasks_.empty())
  {
    task_list::iterator interrupt_task = running_tasks_.begin();
    interrupt_task->task->interrupt_task(interrupt_task->arg);

    // The interrupted task gets moved to the back of the list so that the
    // other running tasks will get interrupted to make way for any new
    // completions that come along.
    running_tasks_.splice(running_tasks_.end(), running_tasks_,
        interrupt_task);

    return true;
  }

  return false;
}

void
shared_thread_demuxer_service::
do_completion_upcall(
    const completion_handler& handler)
  throw ()
{
  try
  {
    handler();
  }
  catch (...)
  {
  }
}

} // namespace detail
} // namespace asio
