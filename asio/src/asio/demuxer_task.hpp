//
// demuxer_task.hpp
// ~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DEMUXER_TASK_HPP
#define ASIO_DEMUXER_TASK_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/thread/xtime.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {

/// The demuxer_task class is a base class for task implementations that
/// provide asynchronous services via the demuxer.
class demuxer_task
{
public:
  /// Destructor.
  virtual ~demuxer_task();

  /// Prepare the task for a subsequent call to execute_task. Since it is
  /// possible for a task to be interrupted before the call to execute_task
  /// begins, initialisation of the interrupt mechanism should occur here. This
  /// function must not make any calls back to the demuxer, as deadlock may
  /// occur.
  virtual void prepare_task(void* arg) throw ();

  /// Execute the task for at most the given length of time. Returns true if
  /// the task has no more work to do and should be removed from the demuxer.
  /// Returns false if the task needs to be executed again.
  virtual bool execute_task(const boost::xtime& interval,
      void* arg) throw () = 0;

  /// Interrupt the task's execution. This function must not make any calls
  /// back to the demuxer, as deadlock may occur.
  virtual void interrupt_task(void* arg) throw ();
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEMUXER_TASK_HPP
