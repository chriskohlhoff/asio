//
// selector.hpp
// ~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SELECTOR_HPP
#define ASIO_DETAIL_SELECTOR_HPP

#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include "asio/demuxer_task.hpp"
#include "asio/detail/select_interrupter.hpp"
#include "asio/detail/select_op_queue.hpp"

#include "asio/detail/push_options.hpp"

namespace asio { class demuxer; }

namespace asio {
namespace detail {

class selector
  : public demuxer_task,
    private boost::noncopyable
{
public:
  // Constructor.
  selector(demuxer& d);

  // Destructor.
  ~selector();

  // Start a new read operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready to be read.
  void start_read_op(select_op& op);

  // Start a read operation from inside an op_call invocation. The do_operation
  // function of the select_op object will be invoked when the given descriptor
  // is ready to be read.
  void restart_read_op(select_op& op);

  // Start a new write operation. The do_operation function of the select_op
  // object will be invoked when the given descriptor is ready for writing.
  void start_write_op(select_op& op);

  // Start a write operation from inside an op_call invocation. The
  // do_operation function of the select_op object will be invoked when the
  // given descriptor is ready for writing.
  void restart_write_op(select_op& op);

  // Close the given descriptor and cancel any operations that are running
  // against it.
  void close_descriptor(socket_type descriptor);

  // Prepare the select loop for execution by resetting the interrupter.
  virtual void prepare_task(void* arg) throw ();

  // Run the select loop.
  virtual bool execute_task(const boost::xtime& interval, void* arg) throw ();

  // Interrupt the select loop.
  virtual void interrupt_task(void* arg) throw ();

private:
  // Mutex to protect access to internal data.
  boost::mutex mutex_;

  // The demuxer that is used to manage selector tasks.
  demuxer& demuxer_;

  // The interrupter is used to break a blocking select call.
  select_interrupter interrupter_;

  // The queue of read operations.
  select_op_queue read_op_queue_;

  // The queue of write operations.
  select_op_queue write_op_queue_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECTOR_HPP
