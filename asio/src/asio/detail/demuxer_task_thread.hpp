//
// demuxer_task_thread.hpp
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

#ifndef ASIO_DETAIL_DEMUXER_TASK_THREAD_HPP
#define ASIO_DETAIL_DEMUXER_TASK_THREAD_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/thread.hpp>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/demuxer_task.hpp"

namespace asio {
namespace detail {

class demuxer_task_thread
  : private boost::noncopyable
{
public:
  // Constructor.
  demuxer_task_thread(demuxer_task& task, void* arg);

  // Destructor.
  ~demuxer_task_thread();

private:
  // Execute the task in a loop.
  void run_task();

  // Mutex to synchronise access to internal data.
  boost::mutex mutex_;

  // The task that is being run in the thread.
  demuxer_task* task_;

  // The argument that is passed to the task's functions.
  void* arg_;

  // Does the task need to be stopped.
  bool stop_;

  // The thread in which the task is being run.
  boost::thread thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_DEMUXER_TASK_THREAD_HPP
