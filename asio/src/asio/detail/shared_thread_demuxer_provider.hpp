//
// shared_thread_demuxer_provider.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SHARED_THREAD_DEMUXER_PROVIDER_HPP
#define ASIO_DETAIL_SHARED_THREAD_DEMUXER_PROVIDER_HPP

#include <list>
#include <queue>
#include <boost/thread.hpp>
#include "asio/completion_context_locker.hpp"
#include "asio/demuxer_service.hpp"
#include "asio/demuxer_task.hpp"
#include "asio/service_provider.hpp"
#include "asio/detail/demuxer_thread_pool.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class shared_thread_demuxer_provider
  : public service_provider,
    public demuxer_service,
    public completion_context_locker
{
public:
  // Constructor.
  shared_thread_demuxer_provider();

  // Destructor.
  virtual ~shared_thread_demuxer_provider();

  // Return the service interface corresponding to the given type.
  virtual service* do_get_service(const service_type_id& service_type);

  // Run the demuxer's event processing loop.
  virtual void run();

  // Interrupt the demuxer's event processing loop.
  virtual void interrupt();

  // Reset the demuxer in preparation for a subsequent run invocation.
  virtual void reset();

  // Add a task to the demuxer.
  virtual void add_task(demuxer_task& task, void* arg);

  // Notify the demuxer that an operation has started.
  virtual void operation_started();

  // Notify the demuxer that an operation has completed.
  virtual void operation_completed(const completion_handler& handler,
      completion_context& context, bool allow_nested_delivery);

  // Notify the demuxer of an operation that started and finished immediately.
  virtual void operation_immediate(const completion_handler& handler,
      completion_context& context, bool allow_nested_delivery);

  // Callback function when a completion context has been acquired.
  virtual void completion_context_acquired(void* arg) throw ();

private:
  // Interrupt all running tasks and idle threads.
  void interrupt_all_threads();

  // Interrupt a single idle thread. Returns true if a thread was interrupted,
  // false if no running thread could be found to interrupt.
  bool interrupt_one_idle_thread();

  // Interrupt a single task. Returns true if a task was interrupted, false if
  // no running task could be found to interrupt.
  bool interrupt_one_task();

  // Structure containing information about a completion to be delivered.
  struct completion_info
  {
    completion_handler handler;
    completion_context* context;
  };

  // Do the upcall for the given completion. This function simply prevents any
  // exceptions from propagating out of the completion handler.
  static void do_completion_upcall(const completion_handler& handler) throw ();

  // Mutex to protect access to internal data.
  boost::mutex mutex_;

  // Structure containing information about a task to be run.
  struct task_info
  {
    demuxer_task* task;
    void* arg;
  };

  // Type of a list of tasks.
  typedef std::list<task_info> task_list;

  // The tasks that are waiting to run.
  task_list waiting_tasks_;

  // The tasks that are currently running.
  task_list running_tasks_;

  // The number of operations that have not yet completed.
  int outstanding_operations_;

  // The type of a queue of completions.
  typedef std::queue<completion_info> completion_queue;

  // The completions that are ready to be delivered.
  completion_queue ready_completions_;

  // Flag to indicate that the dispatcher has been interrupted.
  bool interrupted_;

  // Thread pool to keep track of threads currently inside a run invocation.
  demuxer_thread_pool thread_pool_;

  // The number of threads that are currently idle.
  int idle_thread_count_;

  // Condition variable used by idle threads to wait for interruption.
  boost::condition idle_thread_condition_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SHARED_THREAD_DEMUXER_PROVIDER_HPP
