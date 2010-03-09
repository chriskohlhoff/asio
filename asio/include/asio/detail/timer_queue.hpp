//
// timer_queue.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_TIMER_QUEUE_HPP
#define ASIO_DETAIL_TIMER_QUEUE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstddef>
#include <memory>
#include <vector>
#include <boost/config.hpp>
#include <boost/limits.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/detail/hash_map.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/timer_op.hpp"
#include "asio/detail/timer_queue_base.hpp"

namespace asio {
namespace detail {

template <typename Time_Traits>
class timer_queue
  : public timer_queue_base
{
public:
  // The time type.
  typedef typename Time_Traits::time_type time_type;

  // The duration type.
  typedef typename Time_Traits::duration_type duration_type;

  // Constructor.
  timer_queue()
    : timers_(),
      heap_()
  {
  }

  // Add a new timer to the queue. Returns true if this is the timer that is
  // earliest in the queue, in which case the reactor's event demultiplexing
  // function call may need to be interrupted and restarted.
  bool enqueue_timer(const time_type& time, timer_op* op, void* token)
  {
    // Ensure that there is space for the timer in the heap. We reserve here so
    // that the push_back below will not throw due to a reallocation failure.
    heap_.reserve(heap_.size() + 1);

    // Insert the new timer into the hash.
    typedef typename hash_map<void*, timer>::iterator iterator;
    typedef typename hash_map<void*, timer>::value_type value_type;
    std::pair<iterator, bool> result =
      timers_.insert(value_type(token, timer()));
    result.first->second.op_queue_.push(op);
    if (result.second)
    {
      // Put the new timer at the correct position in the heap.
      result.first->second.time_ = time;
      result.first->second.heap_index_ = heap_.size();
      result.first->second.token_ = token;
      heap_.push_back(&result.first->second);
      up_heap(heap_.size() - 1);
    }

    return (heap_[0] == &result.first->second);
  }

  // Whether there are no timers in the queue.
  virtual bool empty() const
  {
    return heap_.empty();
  }

  // Get the time for the timer that is earliest in the queue.
  virtual long wait_duration_msec(long max_duration) const
  {
    if (heap_.empty())
      return max_duration;

    boost::posix_time::time_duration duration = Time_Traits::to_posix_duration(
        Time_Traits::subtract(heap_[0]->time_, Time_Traits::now()));

    if (duration > boost::posix_time::milliseconds(max_duration))
      duration = boost::posix_time::milliseconds(max_duration);
    else if (duration < boost::posix_time::milliseconds(0))
      duration = boost::posix_time::milliseconds(0);

    return duration.total_milliseconds();
  }

  // Get the time for the timer that is earliest in the queue.
  virtual long wait_duration_usec(long max_duration) const
  {
    if (heap_.empty())
      return max_duration;

    boost::posix_time::time_duration duration = Time_Traits::to_posix_duration(
        Time_Traits::subtract(heap_[0]->time_, Time_Traits::now()));

    if (duration > boost::posix_time::microseconds(max_duration))
      duration = boost::posix_time::microseconds(max_duration);
    else if (duration < boost::posix_time::microseconds(0))
      duration = boost::posix_time::microseconds(0);

    return duration.total_microseconds();
  }

  // Dequeue all timers not later than the current time.
  virtual void get_ready_timers(op_queue<operation>& ops)
  {
    const time_type now = Time_Traits::now();
    while (!heap_.empty() && !Time_Traits::less_than(now, heap_[0]->time_))
    {
      timer* t = heap_[0];
      ops.push(t->op_queue_);
      remove_timer(t);
    }
  }

  // Dequeue all timers.
  virtual void get_all_timers(op_queue<operation>& ops)
  {
    typename hash_map<void*, timer>::iterator i = timers_.begin();
    typename hash_map<void*, timer>::iterator end = timers_.end();
    while (i != end)
    {
      ops.push(i->second.op_queue_);
      typename hash_map<void*, timer>::iterator old_i = i++;
      timers_.erase(old_i);
    }

    heap_.clear();
    timers_.clear();
  }

  // Cancel and dequeue the timers with the given token.
  std::size_t cancel_timer(void* timer_token, op_queue<operation>& ops)
  {
    std::size_t num_cancelled = 0;
    typedef typename hash_map<void*, timer>::iterator iterator;
    iterator it = timers_.find(timer_token);
    if (it != timers_.end())
    {
      while (timer_op* op = it->second.op_queue_.front())
      {
        op->ec_ = asio::error::operation_aborted;
        it->second.op_queue_.pop();
        ops.push(op);
        ++num_cancelled;
      }
      remove_timer(&it->second);
    }
    return num_cancelled;
  }

private:
  // Structure representing a single outstanding timer.
  struct timer
  {
    timer() {}
    timer(const timer&) {}
    void operator=(const timer&) {}

    // The time when the timer should fire.
    time_type time_;

    // The operations waiting on the timer.
    op_queue<timer_op> op_queue_;

    // The index of the timer in the heap.
    size_t heap_index_;

    // The token associated with the timer.
    void* token_;
  };

  // Move the item at the given index up the heap to its correct position.
  void up_heap(size_t index)
  {
    size_t parent = (index - 1) / 2;
    while (index > 0
        && Time_Traits::less_than(heap_[index]->time_, heap_[parent]->time_))
    {
      swap_heap(index, parent);
      index = parent;
      parent = (index - 1) / 2;
    }
  }

  // Move the item at the given index down the heap to its correct position.
  void down_heap(size_t index)
  {
    size_t child = index * 2 + 1;
    while (child < heap_.size())
    {
      size_t min_child = (child + 1 == heap_.size()
          || Time_Traits::less_than(
            heap_[child]->time_, heap_[child + 1]->time_))
        ? child : child + 1;
      if (Time_Traits::less_than(heap_[index]->time_, heap_[min_child]->time_))
        break;
      swap_heap(index, min_child);
      index = min_child;
      child = index * 2 + 1;
    }
  }

  // Swap two entries in the heap.
  void swap_heap(size_t index1, size_t index2)
  {
    timer* tmp = heap_[index1];
    heap_[index1] = heap_[index2];
    heap_[index2] = tmp;
    heap_[index1]->heap_index_ = index1;
    heap_[index2]->heap_index_ = index2;
  }

  // Remove a timer from the heap and list of timers.
  void remove_timer(timer* t)
  {
    // Remove the timer from the heap.
    size_t index = t->heap_index_;
    if (!heap_.empty() && index < heap_.size())
    {
      if (index == heap_.size() - 1)
      {
        heap_.pop_back();
      }
      else
      {
        swap_heap(index, heap_.size() - 1);
        heap_.pop_back();
        size_t parent = (index - 1) / 2;
        if (index > 0 && Time_Traits::less_than(
              heap_[index]->time_, heap_[parent]->time_))
          up_heap(index);
        else
          down_heap(index);
      }
    }

    // Remove the timer from the hash.
    typedef typename hash_map<void*, timer>::iterator iterator;
    iterator it = timers_.find(t->token_);
    if (it != timers_.end())
      timers_.erase(it);
  }

  // A hash of timer token to linked lists of timers.
  hash_map<void*, timer> timers_;

  // The heap of timers, with the earliest timer at the front.
  std::vector<timer*> heap_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_TIMER_QUEUE_HPP
