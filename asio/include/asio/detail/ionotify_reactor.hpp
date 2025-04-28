//
// detail/ionotify_reactor.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IONOTIFY_REACTOR_HPP
#define ASIO_DETAIL_IONOTIFY_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include <asio/detail/config.hpp>

#if defined(ASIO_HAS_IONOTIFY)

#include <cstddef>
#include <asio/detail/fd_set_adapter.hpp>
#include <asio/detail/limits.hpp>
#include <asio/detail/mutex.hpp>
#include <asio/detail/op_queue.hpp>
#include <asio/detail/reactor_op.hpp>
#include <asio/detail/reactor_op_queue.hpp>
#include "asio/detail/scheduler_task.hpp"
#include <asio/detail/socket_types.hpp>
#include <asio/detail/timer_queue_base.hpp>
#include <asio/detail/timer_queue_set.hpp>
#include <asio/detail/wait_op.hpp>
#include <asio/execution_context.hpp>

#include <asio/detail/push_options.hpp>

namespace asio {
namespace detail {

class ionotify_reactor
  : public execution_context_service_base<ionotify_reactor>,
    public scheduler_task
{
public:
  enum op_types { read_op = 0, write_op = 1, except_op = 2,
    max_select_ops = 3, connect_op = 1, max_ops = 3 };

  // Per-descriptor data.
  struct per_descriptor_data
  {
  };

  // Constructor.
  ASIO_DECL ionotify_reactor(asio::execution_context& ctx);

  // Destructor.
  ASIO_DECL ~ionotify_reactor();

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown();

  // Recreate internal descriptors following a fork.
  ASIO_DECL void notify_fork(
      asio::execution_context::fork_event fork_ev);

  // Initialise the task, but only if the reactor is not in its own thread.
  ASIO_DECL void init_task();

  // Register a socket with the reactor. Returns 0 on success, system error
  // code on failure.
  ASIO_DECL int register_descriptor(socket_type, per_descriptor_data&);

  // Register a descriptor with an associated single operation. Returns 0 on
  // success, system error code on failure.
  ASIO_DECL int register_internal_descriptor(
      int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data, reactor_op* op);

  // Post a reactor operation for immediate completion.
  void post_immediate_completion(operation* op, bool is_continuation) const;

  // Post a reactor operation for immediate completion.
  ASIO_DECL static void call_post_immediate_completion(
      operation* op, bool is_continuation, const void* self);

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  ASIO_DECL void start_op(int op_type, socket_type descriptor,
      per_descriptor_data&, reactor_op* op,
      bool is_continuation, bool allow_speculative,
      void (*on_immediate)(operation*, bool, const void*),
      const void* immediate_arg);

  // Start a new operation. The reactor operation will be performed when the
  // given descriptor is flagged as ready, or an error has occurred.
  void start_op(int op_type, socket_type descriptor,
      per_descriptor_data& descriptor_data, reactor_op* op,
      bool is_continuation, bool allow_speculative)
  {
    start_op(op_type, descriptor, descriptor_data,
        op, is_continuation, allow_speculative,
        &ionotify_reactor::call_post_immediate_completion, this);
  }
  
  // Cancel all operations associated with the given descriptor. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  ASIO_DECL void cancel_ops(socket_type descriptor, per_descriptor_data&);

  // Cancel all operations associated with the given descriptor and key. The
  // handlers associated with the descriptor will be invoked with the
  // operation_aborted error.
  ASIO_DECL void cancel_ops_by_key(socket_type descriptor,
      per_descriptor_data& descriptor_data,
      int op_type, void* cancellation_key);

  // Cancel any operations that are running against the descriptor and remove
  // its registration from the reactor. The reactor resources associated with
  // the descriptor must be released by calling cleanup_descriptor_data.
  ASIO_DECL void deregister_descriptor(socket_type descriptor,
      per_descriptor_data&, bool closing);

  // Remove the descriptor's registration from the reactor. The reactor
  // resources associated with the descriptor must be released by calling
  // cleanup_descriptor_data.
  ASIO_DECL void deregister_internal_descriptor(
      socket_type descriptor, per_descriptor_data&);

  // Perform any post-deregistration cleanup tasks associated with the
  // descriptor data.
  ASIO_DECL void cleanup_descriptor_data(per_descriptor_data&);

  // Move descriptor registration from one descriptor_data object to another.
  ASIO_DECL void move_descriptor(socket_type descriptor,
      per_descriptor_data& target_descriptor_data,
      per_descriptor_data& source_descriptor_data);

  // Add a new timer queue to the reactor.
  template <typename Time_Traits>
  void add_timer_queue(timer_queue<Time_Traits>& queue);

  // Remove a timer queue from the reactor.
  template <typename Time_Traits>
  void remove_timer_queue(timer_queue<Time_Traits>& queue);

  // Schedule a new operation in the given timer queue to expire at the
  // specified absolute time.
  template <typename Time_Traits>
  void schedule_timer(timer_queue<Time_Traits>& queue,
      const typename Time_Traits::time_type& time,
      typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op);

  // Cancel the timer operations associated with the given token. Returns the
  // number of operations that have been posted or dispatched.
  template <typename Time_Traits>
  std::size_t cancel_timer(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data& timer,
      std::size_t max_cancelled = (std::numeric_limits<std::size_t>::max)());

  // Cancel the timer operations associated with the given key.
  template <typename Time_Traits>
  void cancel_timer_by_key(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data* timer,
      void* cancellation_key);

  // Move the timer operations associated with the given timer.
  template <typename Time_Traits>
  void move_timer(timer_queue<Time_Traits>& queue,
      typename timer_queue<Time_Traits>::per_timer_data& target,
      typename timer_queue<Time_Traits>::per_timer_data& source);

  // Run select once until interrupted or events are ready to be dispatched.
  ASIO_DECL void run(long usec, op_queue<operation>& ops);

  // Interrupt the select loop.
  ASIO_DECL void interrupt();

private:
  // Helper function that acts like socket_ops::select() with our fd_sets, but internally uses pulses.
  ASIO_DECL int select(asio::detail::mutex::scoped_lock& lock, socket_type max_fd, long timeout_usec);

  // Helper function to add a new timer queue.
  ASIO_DECL void do_add_timer_queue(timer_queue_base& queue);

  // Helper function to remove a timer queue.
  ASIO_DECL void do_remove_timer_queue(timer_queue_base& queue);

  // Get the timeout value for the select call.
  ASIO_DECL long get_timeout_usec(long usec);

  // Cancel all operations associated with the given descriptor. This function
  // does not acquire the ionotify_reactor's mutex.
  ASIO_DECL void cancel_ops_unlocked(socket_type descriptor,
      const asio::error_code& ec);

  // Deregister the given descriptor.  This function
  // does not acquire the ionotify_reactor's mutex.
  ASIO_DECL void deregister_descriptor_unlocked(socket_type descriptor);

  // Internal version of interrupt(), called with the mutex already locked
  ASIO_DECL void interrupt_unlocked();

  // Create pulse channel for select()
  ASIO_DECL void create_pulse_channel(void);

  // The scheduler implementation used to post completions.
  typedef class scheduler scheduler_type;
  scheduler_type& scheduler_;

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The queues of read, write and except operations.
  reactor_op_queue<socket_type> op_queue_[max_ops];

  // Structure that holds the state of an fd and what we want from it.
  // (We'll have a vector of these.)
  struct fdstate
  {
    unsigned  ops_     : max_select_ops;
    bool      wanted_  : 1;
    unsigned  armed_   : max_select_ops;
    sigevent  ioev_;

    enum op_bits
    {
      NONE     = 0,
      READ     = 1 << read_op,
      WRITE    = 1 << write_op,
      EXCEPT   = 1 << except_op,
      ALL      = READ | WRITE | EXCEPT
    };
    fdstate() : ops_(NONE), wanted_(false), armed_(NONE) {ioev_.sigev_notify = 0;}
  };
  using op_bits = fdstate::op_bits;
  static op_bits op_bit(op_types op) { return static_cast<op_bits>(1 << op); }

  class fdmap: public std::vector<fdstate>
  {
  public:
    // Similar to posix_fd_set_adapter::set but returns the max fd (or invalid_socket)
    ASIO_DECL socket_type set(reactor_op_queue<socket_type>& operations, op_queue<operation>& ops, op_types op);
    // Similar to posix_fd_set_adapter::perform but returns the max fd (or invalid_socket)
    ASIO_DECL void perform(reactor_op_queue<socket_type>& operations, op_queue<operation>& ops, op_types op);
    bool is_valid(socket_type fd) { return fd >= 0 && static_cast<size_t>(fd) < size(); }
  } fdmap_;

  // The timer queues.
  timer_queue_set timer_queues_;

  // The channel and coid used for pulses
  int chid_, coid_;
  // Whether we have already sent an interrupt pulse
  bool interrupted_;

  // Whether the service has been shut down.
  bool shutdown_;
};

} // namespace detail
} // namespace asio

#include <asio/detail/pop_options.hpp>

#include <asio/detail/impl/ionotify_reactor.hpp>
#if defined(ASIO_HEADER_ONLY)
# include <asio/detail/impl/ionotify_reactor.ipp>
#endif // defined(ASIO_HEADER_ONLY)

#endif // defined(ASIO_HAS_IONOTIFY)

#endif // ASIO_DETAIL_IONOTIFY_REACTOR_HPP
