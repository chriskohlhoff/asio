//
// reactor_op_queue.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_REACTOR_OP_QUEUE_HPP
#define ASIO_DETAIL_REACTOR_OP_QUEUE_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/hash_map.hpp"

namespace asio {
namespace detail {

template <typename Descriptor>
class reactor_op_queue
  : private boost::noncopyable
{
public:
  // Constructor.
  reactor_op_queue()
    : operations_(),
      cancelled_operations_(0)
  {
  }

  // Add a new operation to the queue. Returns true if this is the only
  // operation for the given descriptor, in which case the reactor's event
  // demultiplexing function call may need to be interrupted and restarted.
  template <typename Handler>
  bool enqueue_operation(Descriptor descriptor, Handler handler)
  {
    op_base* new_op = new op<Handler>(descriptor, handler);

    typedef typename operation_map::iterator iterator;
    typedef typename operation_map::value_type value_type;
    std::pair<iterator, bool> entry =
      operations_.insert(value_type(descriptor, new_op));
    if (entry.second)
      return true;

    op_base* current_op = entry.first->second;
    while (current_op->next_)
      current_op = current_op->next_;
    current_op->next_ = new_op;

    return false;
  }

  // Close the given descriptor. Any operations pending for the descriptor will
  // be notified that they have been cancelled next time dispatch_cancellations
  // is called. Returns true if any operations were cancelled, in which case
  // the reactor's event demultiplexing function may need to be interrupted and
  // restarted.
  bool close_descriptor(Descriptor descriptor)
  {
    typename operation_map::iterator i = operations_.find(descriptor);
    if (i != operations_.end())
    {
      op_base* last_op = i->second;
      while (last_op->next_)
        last_op = last_op->next_;
      last_op->next_ = cancelled_operations_;
      cancelled_operations_ = i->second;
      operations_.erase(i);
      return true;
    }

    return false;
  }

  // Whether there are no operations in the queue.
  bool empty() const
  {
    return operations_.empty();
  }

  // Fill a descriptor set with the descriptors corresponding to each active
  // operation.
  template <typename Descriptor_Set>
  void get_descriptors(Descriptor_Set& descriptors)
  {
    typename operation_map::iterator i = operations_.begin();
    while (i != operations_.end())
    {
      descriptors.set(i->first);
      ++i;
    }
  }

  // Dispatch the operations corresponding to the ready file descriptors
  // contained in the given descriptor set.
  template <typename Descriptor_Set>
  void dispatch_descriptors(const Descriptor_Set& descriptors)
  {
    typename operation_map::iterator i = operations_.begin();
    while (i != operations_.end())
    {
      typename operation_map::iterator op = i++;
      if (descriptors.is_set(op->first))
      {
        op_base* next_op = op->second->next_;
        op->second->next_ = 0;
        op->second->do_operation();
        delete op->second;
        if (next_op)
          op->second = next_op;
        else
          operations_.erase(op);
      }
    }
  }

  // Dispatch any pending cancels for operations.
  void dispatch_cancellations()
  {
    while (cancelled_operations_)
    {
      op_base* next_op = cancelled_operations_->next_;
      cancelled_operations_->next_ = 0;
      cancelled_operations_->do_cancel();
      delete cancelled_operations_;
      cancelled_operations_ = next_op;
    }
  }

private:
  // Base class for reactor operations.
  class op_base
  {
  public:
    // Destructor.
    virtual ~op_base()
    {
    }

    // Get the descriptor associated with the operation.
    Descriptor descriptor() const
    {
      return descriptor_;
    }

    // Perform the operation.
    virtual void do_operation() = 0;

    // Handle the case where the operation has been cancelled.
    virtual void do_cancel() = 0;

  protected:
    // Construct an operation for the given descriptor.
    op_base(Descriptor descriptor)
      : descriptor_(descriptor),
        next_(0)
    {
    }

  private:
    friend class reactor_op_queue<Descriptor>;

    // The descriptor associated with the operation.
    Descriptor descriptor_;

    // The next operation for the same file descriptor.
    op_base* next_;
  };

  // Adaptor class template for using handlers in operations.
  template <typename Handler>
  class op
    : public op_base
  {
  public:
    // Constructor.
    op(Descriptor descriptor, Handler handler)
      : op_base(descriptor),
        handler_(handler)
    {
    }

    // Perform the operation.
    virtual void do_operation()
    {
      handler_.do_operation();
    }

    // Handle the case where the operation has been cancelled.
    virtual void do_cancel()
    {
      handler_.do_cancel();
    }

  private:
    Handler handler_;
  };

  // The type for a map of operations.
  typedef hash_map<Descriptor, op_base*> operation_map;

  // The operations that are currently executing asynchronously.
  operation_map operations_;

  // The list of operations that have been cancelled.
  op_base* cancelled_operations_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTOR_OP_QUEUE_HPP
