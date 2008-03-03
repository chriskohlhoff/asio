//
// reactor_op_queue.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTOR_OP_QUEUE_HPP
#define ASIO_DETAIL_REACTOR_OP_QUEUE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/detail/hash_map.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

template <typename Descriptor>
class reactor_op_queue
  : private noncopyable
{
public:
  // Constructor.
  reactor_op_queue()
    : operations_(),
      cancelled_operations_(0),
      cleanup_operations_(0)
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

  // Cancel all operations associated with the descriptor. Any operations
  // pending for the descriptor will be notified that they have been cancelled
  // next time dispatch_cancellations is called. Returns true if any operations
  // were cancelled, in which case the reactor's event demultiplexing function
  // may need to be interrupted and restarted.
  bool cancel_operations(Descriptor descriptor)
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

  // Determine whether there are any operations associated with the descriptor.
  bool has_operation(Descriptor descriptor) const
  {
    return operations_.find(descriptor) != operations_.end();
  }

  // Dispatch the first operation corresponding to the descriptor. Returns true
  // if there are more operations queued for the descriptor.
  bool dispatch_operation(Descriptor descriptor,
      const asio::error_code& result)
  {
    typename operation_map::iterator i = operations_.find(descriptor);
    if (i != operations_.end())
    {
      op_base* this_op = i->second;
      i->second = this_op->next_;
      this_op->next_ = cleanup_operations_;
      cleanup_operations_ = this_op;
      bool done = this_op->invoke(result);
      if (done)
      {
        // Operation has finished.
        if (i->second)
        {
          return true;
        }
        else
        {
          operations_.erase(i);
          return false;
        }
      }
      else
      {
        // Operation wants to be called again. Leave it at the front of the
        // queue for this descriptor, and remove from the cleanup list.
        cleanup_operations_ = this_op->next_;
        this_op->next_ = i->second;
        i->second = this_op;
        return true;
      }
    }
    return false;
  }

  // Dispatch all operations corresponding to the descriptor.
  void dispatch_all_operations(Descriptor descriptor,
      const asio::error_code& result)
  {
    typename operation_map::iterator i = operations_.find(descriptor);
    if (i != operations_.end())
    {
      while (i->second)
      {
        op_base* this_op = i->second;
        i->second = this_op->next_;
        this_op->next_ = cleanup_operations_;
        cleanup_operations_ = this_op;
        bool done = this_op->invoke(result);
        if (!done)
        {
          // Operation has not finished yet, so leave at front of queue, and
          // remove from the cleanup list.
          cleanup_operations_ = this_op->next_;
          this_op->next_ = i->second;
          i->second = this_op;
          return;
        }
      }
      operations_.erase(i);
    }
  }

  // Fill a descriptor set with the descriptors corresponding to each active
  // operation.
  template <typename Descriptor_Set>
  void get_descriptors(Descriptor_Set& descriptors)
  {
    typename operation_map::iterator i = operations_.begin();
    while (i != operations_.end())
    {
      Descriptor descriptor = i->first;
      ++i;
      if (!descriptors.set(descriptor))
      {
        asio::error_code ec(error::fd_set_failure);
        dispatch_all_operations(descriptor, ec);
      }
    }
  }

  // Dispatch the operations corresponding to the ready file descriptors
  // contained in the given descriptor set.
  template <typename Descriptor_Set>
  void dispatch_descriptors(const Descriptor_Set& descriptors,
      const asio::error_code& result)
  {
    typename operation_map::iterator i = operations_.begin();
    while (i != operations_.end())
    {
      typename operation_map::iterator op_iter = i++;
      if (descriptors.is_set(op_iter->first))
      {
        op_base* this_op = op_iter->second;
        op_iter->second = this_op->next_;
        this_op->next_ = cleanup_operations_;
        cleanup_operations_ = this_op;
        bool done = this_op->invoke(result);
        if (done)
        {
          if (!op_iter->second)
            operations_.erase(op_iter);
        }
        else
        {
          // Operation has not finished yet, so leave at front of queue, and
          // remove from the cleanup list.
          cleanup_operations_ = this_op->next_;
          this_op->next_ = op_iter->second;
          op_iter->second = this_op;
        }
      }
    }
  }

  // Dispatch any pending cancels for operations.
  void dispatch_cancellations()
  {
    while (cancelled_operations_)
    {
      op_base* this_op = cancelled_operations_;
      cancelled_operations_ = this_op->next_;
      this_op->next_ = cleanup_operations_;
      cleanup_operations_ = this_op;
      this_op->invoke(asio::error::operation_aborted);
    }
  }

  // Destroy operations that are waiting to be cleaned up.
  void cleanup_operations()
  {
    while (cleanup_operations_)
    {
      op_base* next_op = cleanup_operations_->next_;
      cleanup_operations_->next_ = 0;
      cleanup_operations_->destroy();
      cleanup_operations_ = next_op;
    }
  }

  // Destroy all operations owned by the queue.
  void destroy_operations()
  {
    while (cancelled_operations_)
    {
      op_base* next_op = cancelled_operations_->next_;
      cancelled_operations_->next_ = 0;
      cancelled_operations_->destroy();
      cancelled_operations_ = next_op;
    }

    while (cleanup_operations_)
    {
      op_base* next_op = cleanup_operations_->next_;
      cleanup_operations_->next_ = 0;
      cleanup_operations_->destroy();
      cleanup_operations_ = next_op;
    }

    typename operation_map::iterator i = operations_.begin();
    while (i != operations_.end())
    {
      typename operation_map::iterator op_iter = i++;
      op_base* curr_op = op_iter->second;
      operations_.erase(op_iter);
      while (curr_op)
      {
        op_base* next_op = curr_op->next_;
        curr_op->next_ = 0;
        curr_op->destroy();
        curr_op = next_op;
      }
    }
  }

private:
  // Base class for reactor operations. A function pointer is used instead of
  // virtual functions to avoid the associated overhead.
  class op_base
  {
  public:
    // Get the descriptor associated with the operation.
    Descriptor descriptor() const
    {
      return descriptor_;
    }

    // Perform the operation.
    bool invoke(const asio::error_code& result)
    {
      return invoke_func_(this, result);
    }

    // Destroy the operation.
    void destroy()
    {
      return destroy_func_(this);
    }

  protected:
    typedef bool (*invoke_func_type)(op_base*,
        const asio::error_code&);
    typedef void (*destroy_func_type)(op_base*);

    // Construct an operation for the given descriptor.
    op_base(invoke_func_type invoke_func,
        destroy_func_type destroy_func, Descriptor descriptor)
      : invoke_func_(invoke_func),
        destroy_func_(destroy_func),
        descriptor_(descriptor),
        next_(0)
    {
    }

    // Prevent deletion through this type.
    ~op_base()
    {
    }

  private:
    friend class reactor_op_queue<Descriptor>;

    // The function to be called to dispatch the handler.
    invoke_func_type invoke_func_;

    // The function to be called to delete the handler.
    destroy_func_type destroy_func_;

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
      : op_base(&op<Handler>::invoke_handler,
          &op<Handler>::destroy_handler, descriptor),
        handler_(handler)
    {
    }

    // Invoke the handler.
    static bool invoke_handler(op_base* base,
        const asio::error_code& result)
    {
      return static_cast<op<Handler>*>(base)->handler_(result);
    }

    // Delete the handler.
    static void destroy_handler(op_base* base)
    {
      delete static_cast<op<Handler>*>(base);
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

  // The list of operations to be destroyed.
  op_base* cleanup_operations_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTOR_OP_QUEUE_HPP
