//
// task_demuxer_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
      outstanding_work_(0),
      handler_queue_(0),
      handler_queue_end_(0),
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

    while (!interrupted_ && outstanding_work_ > 0)
    {
      if (handler_queue_)
      {
        handler_base* h = handler_queue_;
        handler_queue_ = h->next_;
        if (handler_queue_ == 0)
          handler_queue_end_ = 0;
        lock.unlock();
        h->call(); // call() deletes the handler object
        lock.lock();
        --outstanding_work_;
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

  // Notify the demuxer that some work has started.
  void work_started()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    ++outstanding_work_;
  }

  // Notify the demuxer that some work has finished.
  void work_finished()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (--outstanding_work_ == 0)
      interrupt_all_threads();
  }

  // Request the demuxer to invoke the given handler.
  template <typename Handler>
  void dispatch(Handler handler)
  {
    if (current_thread_in_pool_)
      handler_wrapper<Handler>::do_upcall(handler);
    else
      post(handler);
  }

  // Request the demuxer to invoke the given handler and return immediately.
  template <typename Handler>
  void post(Handler handler)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Add the handler to the end of the queue.
    handler_base* h = new handler_wrapper<Handler>(handler);
    if (handler_queue_end_)
    {
      handler_queue_end_->next_ = h;
      handler_queue_end_ = h;
    }
    else
    {
      handler_queue_ = handler_queue_end_ = h;
    }

    // An undelivered handler is treated as unfinished work.
    ++outstanding_work_;

    // Wake up a thread to execute the handler.
    if (!interrupt_one_idle_thread())
      interrupt_task();
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

  // The base class for all handler wrappers. A function pointer is used
  // instead of virtual functions to avoid the associated overhead.
  class handler_base
  {
  public:
    typedef void (*func_type)(handler_base*);

    handler_base(func_type func)
      : next_(0),
        func_(func)
    {
    }

    void call()
    {
      func_(this);
    }

  protected:
    // Prevent deletion through this type.
    ~handler_base()
    {
    }

  private:
    friend class task_demuxer_service<Task>;
    handler_base* next_;
    func_type func_;
  };

  // Template wrapper for handlers.
  template <typename Handler>
  class handler_wrapper
    : public handler_base
  {
  public:
    handler_wrapper(Handler handler)
      : handler_base(&handler_wrapper<Handler>::do_call),
        handler_(handler)
    {
    }

    static void do_call(handler_base* base)
    {
      handler_wrapper<Handler>* h =
        static_cast<handler_wrapper<Handler>*>(base);
      h->do_upcall(h->handler_);
      delete h;
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
  };

  // The demuxer that owns this service.
  demuxer_type& demuxer_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The task to be run by this demuxer service.
  Task& task_;

  // Whether the task is currently running.
  bool task_is_running_;

  // The count of unfinished work.
  int outstanding_work_;

  // The start of a linked list of handlers that are ready to be delivered.
  handler_base* handler_queue_;

  // The end of a linked list of handlers that are ready to be delivered.
  handler_base* handler_queue_end_;

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
