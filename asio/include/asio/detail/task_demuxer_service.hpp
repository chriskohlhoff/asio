//
// task_demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_TASK_DEMUXER_SERVICE_HPP
#define ASIO_DETAIL_TASK_DEMUXER_SERVICE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/basic_demuxer.hpp"
#include "asio/service_factory.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/event.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/tss_bool.hpp"

namespace asio {
namespace detail {

template <typename Task>
class task_demuxer_service
{
public:
  // The demuxer type for this service.
  typedef basic_demuxer<task_demuxer_service<Task> > demuxer_type;

  // Constructor. Taking a reference to the demuxer type as the parameter
  // forces the compiler to ensure that this class can only be used as the
  // demuxer service. It cannot be instantiated in the demuxer in any other
  // case.
  task_demuxer_service(
      demuxer_type& demuxer)
    : demuxer_(demuxer),
      mutex_(),
      task_(demuxer.get_service(service_factory<Task>())),
      task_is_running_(false),
      outstanding_operations_(0),
      ready_completions_(0),
      ready_completions_end_(0),
      interrupted_(false),
      current_thread_in_pool_(),
      first_idle_thread_(0)
  {
  }

  // Get the demuxer associated with the service.
  demuxer_type& demuxer()
  {
    return demuxer_;
  }

  // Create a new dgram socket implementation.
  // Run the demuxer's event processing loop.
  void run()
  {
    current_thread_in_pool_ = true;

    idle_thread_info this_idle_thread;
    this_idle_thread.prev = &this_idle_thread;
    this_idle_thread.next = &this_idle_thread;

    asio::detail::mutex::scoped_lock lock(mutex_);

    while (!interrupted_ && outstanding_operations_ > 0)
    {
      if (ready_completions_)
      {
        completion_base* comp = ready_completions_;
        ready_completions_ = comp->next_;
        if (ready_completions_ == 0)
          ready_completions_end_ = 0;
        lock.unlock();
        comp->call();
        delete comp;
        lock.lock();
        --outstanding_operations_;
      }
      else if (!task_is_running_)
      {
        task_is_running_ = true;
        task_.reset();
        lock.unlock();
        task_.run();
        lock.lock();
        task_is_running_ = false;
      }
      else 
      {
        // Nothing to run right now, so just wait for work to do.
        if (first_idle_thread_)
        {
          this_idle_thread.next = first_idle_thread_;
          this_idle_thread.prev = first_idle_thread_->prev;
          first_idle_thread_->prev->next = &this_idle_thread;
          first_idle_thread_->prev = &this_idle_thread;
        }
        first_idle_thread_ = &this_idle_thread;
        this_idle_thread.wakeup_event.clear();
        lock.unlock();
        this_idle_thread.wakeup_event.wait();
        lock.lock();
        if (this_idle_thread.next == &this_idle_thread)
        {
          first_idle_thread_ = 0;
        }
        else
        {
          if (first_idle_thread_ == &this_idle_thread)
            first_idle_thread_ = this_idle_thread.next;
          this_idle_thread.next->prev = this_idle_thread.prev;
          this_idle_thread.prev->next = this_idle_thread.next;
          this_idle_thread.next = &this_idle_thread;
          this_idle_thread.prev = &this_idle_thread;
        }
      }
    }

    if (!interrupted_)
    {
      // No more work to do!
      interrupt_all_threads();
    }

    current_thread_in_pool_ = false;
  }

  // Interrupt the demuxer's event processing loop.
  void interrupt()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    interrupt_all_threads();
  }

  // Reset the demuxer in preparation for a subsequent run invocation.
  void reset()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    interrupted_ = false;
  }

  // Notify the demuxer that an operation has started.
  void operation_started()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    ++outstanding_operations_;
  }

  // Notify the demuxer that an operation has completed.
  template <typename Handler, typename Completion_Context>
  void operation_completed(Handler handler, Completion_Context context,
      bool allow_nested_delivery)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    if (context.try_acquire())
    {
      if (allow_nested_delivery && current_thread_in_pool_)
      {
        lock.unlock();
        completion<Handler, Completion_Context>::do_upcall(handler);
        context.release();
        lock.lock();
        if (--outstanding_operations_ == 0)
          interrupt_all_threads();
      }
      else
      {
        completion_base* comp =
          new completion<Handler, Completion_Context>(handler, context);
        if (ready_completions_end_)
        {
          ready_completions_end_->next_ = comp;
          ready_completions_end_ = comp;
        }
        else
        {
          ready_completions_ = ready_completions_end_ = comp;
        }
        if (!interrupt_one_idle_thread())
          interrupt_task();
      }
    }
    else
    {
      completion_base* comp =
        new completion<Handler, Completion_Context>(handler, context);
      context.acquire(bind_handler(
            task_demuxer_service<Task>::completion_context_acquired, this,
            comp));
    }
  }

  // Notify the demuxer of an operation that started and finished immediately.
  template <typename Handler, typename Completion_Context>
  void operation_immediate(Handler handler, Completion_Context context,
      bool allow_nested_delivery)
  {
    operation_started();
    operation_completed(handler, context, allow_nested_delivery);
  }

  // Callback function when a completion context has been acquired.
  static void completion_context_acquired(task_demuxer_service<Task>* service,
      void* arg) throw ()
  {
    try
    {
      asio::detail::mutex::scoped_lock lock(service->mutex_);
      completion_base* comp = static_cast<completion_base*>(arg);
      if (service->ready_completions_end_)
      {
        service->ready_completions_end_->next_ = comp;
        service->ready_completions_end_ = comp;
      }
      else
      {
        service->ready_completions_ = service->ready_completions_end_ = comp;
      }
      if (!service->interrupt_one_idle_thread())
        service->interrupt_task();
    }
    catch (...)
    {
    }
  }

private:
  // Interrupt the task and all idle threads.
  void interrupt_all_threads()
  {
    interrupted_ = true;
    interrupt_all_idle_threads();
    interrupt_task();
  }

  // Interrupt a single idle thread. Returns true if a thread was interrupted,
  // false if no running thread could be found to interrupt.
  bool interrupt_one_idle_thread()
  {
    if (first_idle_thread_)
    {
      first_idle_thread_->wakeup_event.signal();
      first_idle_thread_ = first_idle_thread_->next;
      return true;
    }
    return false;
  }

  // Interrupt all idle threads.
  void interrupt_all_idle_threads()
  {
    if (first_idle_thread_)
    {
      first_idle_thread_->wakeup_event.signal();
      idle_thread_info* current_idle_thread = first_idle_thread_->next;
      while (current_idle_thread != first_idle_thread_)
      {
        current_idle_thread->wakeup_event.signal();
        current_idle_thread = current_idle_thread->next;
      }
    }
  }

  // Interrupt the task. Returns true if the task was interrupted, false if
  // the task was not running and so could not be interrupted.
  bool interrupt_task()
  {
    if (task_is_running_)
    {
      task_.interrupt();
      return true;
    }
    return false;
  }

  // The base class for all completions.
  class completion_base
  {
  public:
    virtual ~completion_base()
    {
    }

    virtual void call() = 0;

  protected:
    completion_base()
      : next_(0)
    {
    }

  private:
    friend class task_demuxer_service<Task>;
    completion_base* next_;
  };

  // Template for completions specific to a handler.
  template <typename Handler, typename Completion_Context>
  class completion
    : public completion_base
  {
  public:
    completion(Handler handler, Completion_Context context)
      : handler_(handler),
        context_(context)
    {
    }

    virtual void call()
    {
      do_upcall(handler_);
      context_.release();
    }

    static void do_upcall(Handler& handler)
    {
      try
      {
        handler();
      }
      catch (...)
      {
      }
    }

  private:
    Handler handler_;
    Completion_Context context_;
  };

  // The demuxer that owns this service.
  demuxer_type& demuxer_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The task to be run by this demuxer service.
  Task& task_;

  // Whether the task is currently running.
  bool task_is_running_;

  // The number of operations that have not yet completed.
  int outstanding_operations_;

  // The start of a linked list of completions that are ready to be delivered.
  completion_base* ready_completions_;

  // The end of a linked list of completions that are ready to be delivered.
  completion_base* ready_completions_end_;

  // Flag to indicate that the dispatcher has been interrupted.
  bool interrupted_;

  // Thread-specific flag to keep track of which threads are in the pool.
  tss_bool current_thread_in_pool_;

  // Structure containing information about an idle thread.
  struct idle_thread_info
  {
    event wakeup_event;
    idle_thread_info* prev;
    idle_thread_info* next;
  };

  // The number of threads that are currently idle.
  idle_thread_info* first_idle_thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TASK_DEMUXER_SERVICE_HPP
