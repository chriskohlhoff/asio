//
// select_op_queue.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_SELECT_OP_QUEUE_HPP
#define ASIO_DETAIL_SELECT_OP_QUEUE_HPP

#include <map>
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class select_op;

class select_op_queue
{
public:
  // Constructor.
  select_op_queue();

  // Destructor.
  ~select_op_queue();

  // Add a new select operation to the queue. Returns true if this is the only
  // operation for the given file descriptor, in which case the select call may
  // need to be interrupted and restarted.
  bool enqueue_operation(select_op& op);

  // Whether there are no operations in the queue.
  bool empty() const;

  // Fill a file descriptor set with the descriptors corresponding to each
  // active operation. Returns the highest descriptor added to the set.
  socket_type get_descriptors(fd_set& fds);

  // Dispatch the operations corresponding to the ready file descriptors
  // contained in the given descriptor set.
  void dispatch_descriptors(const fd_set& fds);

  // Close the given descriptor. Any operations pending for the descriptor will
  // be notified that they are being cancelled.
  void close_descriptor(socket_type descriptor);

private:
  // The type for a map of operations.
  typedef std::map<socket_type, select_op*> operation_map;

  // The operations that are currently executing asynchronously.
  operation_map operations_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SELECT_OP_QUEUE_HPP
