//
// select_op_queue.cpp
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

#include "asio/detail/select_op_queue.hpp"
#include <cassert>
#include "asio/detail/select_op.hpp"

namespace asio {
namespace detail {

select_op_queue::
select_op_queue()
  : operations_()
{
}

select_op_queue::
~select_op_queue()
{
}

bool
select_op_queue::
enqueue_operation(
    select_op& op)
{
  assert(op.next_ == 0);

  std::pair<operation_map::iterator, bool> entry =
    operations_.insert(operation_map::value_type(op.descriptor(), &op));
  if (entry.second)
    return true;

  select_op* current_op = entry.first->second;
  while (current_op->next_)
    current_op = current_op->next_;
  current_op->next_ = &op;

  return false;
}

bool
select_op_queue::
empty() const
{
  return operations_.empty();
}

socket_type
select_op_queue::
get_descriptors(
    fd_set& fds)
{
  // Add all active read operations' descriptors into the fd_set.
  socket_type max_fd = -1;
  operation_map::iterator i = operations_.begin();
  while (i != operations_.end())
  {
    FD_SET(i->first, &fds);
    if (i->first > max_fd)
      max_fd = i->first;
    ++i;
  }

  return max_fd;
}

void
select_op_queue::
dispatch_descriptors(
    const fd_set& fds)
{
  // Dispatch all ready operations.
  operation_map::iterator i = operations_.begin();
  while (i != operations_.end())
  {
    operation_map::iterator op = i++;
    if (FD_ISSET(op->first, &fds))
    {
      select_op* next_op = op->second->next_;
      op->second->next_ = 0;
      op->second->do_operation();
      if (next_op)
        op->second = next_op;
      else
        operations_.erase(op);
    }
  }
}

void
select_op_queue::
close_descriptor(
    socket_type descriptor)
{
  operation_map::iterator i = operations_.find(descriptor);
  if (i != operations_.end())
  {
    select_op* op = i->second;
    while (op)
    {
      select_op* next_op = op->next_;
      op->next_ = 0;
      op->do_cancel();
      op = next_op;
    }
    operations_.erase(i);
  }
}

} // namespace detail
} // namespace asio
