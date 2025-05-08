// detail/impl/ionotify_reactor.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_IONOTIFY_REACTOR_IPP
#define ASIO_DETAIL_IMPL_IONOTIFY_REACTOR_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
#pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include <asio/detail/config.hpp>

#if defined(ASIO_HAS_IONOTIFY)

#include <sys/iomsg.h>
#include <sys/trace.h>
#include <sys/neutrino.h>

#include <asio/detail/fd_set_adapter.hpp>
#include <asio/detail/ionotify_reactor.hpp>
#include "asio/detail/scheduler.hpp"
#include <asio/detail/signal_blocker.hpp>
#include <asio/detail/socket_ops.hpp>

#include <asio/detail/push_options.hpp>

#define PULSE_PRIO                                                             \
  10 // We want all our pulses to have the same priority, to preserve ordering
#define INTERRUPT_PULSE_CODE _PULSE_CODE_MINAVAIL
#define SERIAL_BIT 1 // Used to detect old pulses

namespace asio
{
  namespace detail
  {
    ionotify_reactor::ionotify_reactor(asio::execution_context& ctx)
        : execution_context_service_base<ionotify_reactor>(ctx),
          scheduler_(use_service<scheduler_type>(ctx)), mutex_(), chid_(-1),
          coid_(-1), interrupted_(false), shutdown_(false)
    {
      create_pulse_channel();
    }

    ionotify_reactor::~ionotify_reactor()
    {
      shutdown();

      // Unregister all registered events
      for (size_t i = 0; i < fdmap_.size(); i++)
        if (fdmap_[i].ioev_.sigev_notify != 0)
          MsgUnregisterEvent(&fdmap_[i].ioev_);

      if (chid_ >= 0)
      {
        ChannelDestroy(chid_);
        if (coid_ >= 0)
        {
          ConnectDetach(coid_);
        }
      }
    }

    void ionotify_reactor::shutdown()
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      shutdown_ = true;
      lock.unlock();

      op_queue<operation> ops;

      for (int i = 0; i < max_ops; ++i)
        op_queue_[i].get_all_operations(ops);

      timer_queues_.get_all_timers(ops);

      scheduler_.abandon_operations(ops);
    }

    void
    ionotify_reactor::notify_fork(asio::execution_context::fork_event fork_ev)
    {
      if (fork_ev == asio::execution_context::fork_child)
      {
        // Cleanup all registered events
        for (size_t i = 0; i < fdmap_.size(); i++)
        {
          fdmap_[i].ioev_.sigev_notify = 0;
          fdmap_[i].armed_ = 0;
        }

        // Re-create pulse channel for forked process
        chid_ = coid_ = -1;
        create_pulse_channel();
      }
    }

    void ionotify_reactor::init_task()
    {
      scheduler_.init_task();
    }

    int ionotify_reactor::register_descriptor(
        socket_type /*fd*/, ionotify_reactor::per_descriptor_data& /*d*/)
    {
      return 0;
    }

    int ionotify_reactor::register_internal_descriptor(
        int op_type, socket_type descriptor,
        ionotify_reactor::per_descriptor_data& /*d*/, reactor_op* op)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);

      op_queue_[op_type].enqueue_operation(descriptor, op);
      interrupt_unlocked();

      return 0;
    }

    void ionotify_reactor::move_descriptor(
        socket_type /*fd*/, ionotify_reactor::per_descriptor_data& /*d1*/,
        ionotify_reactor::per_descriptor_data& /*d2*/)
    {
    }

    void ionotify_reactor::call_post_immediate_completion(operation* op,
                                                          bool is_continuation,
                                                          const void* self)
    {
      static_cast<const ionotify_reactor*>(self)->post_immediate_completion(
          op, is_continuation);
    }

    void ionotify_reactor::start_op(int op_type, socket_type descriptor,
                                    ionotify_reactor::per_descriptor_data&,
                                    reactor_op* op, bool is_continuation, bool,
                                    void (*on_immediate)(operation*, bool,
                                                         const void*),
                                    const void* immediate_arg)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);

      if (shutdown_)
      {
        on_immediate(op, is_continuation, immediate_arg);
        return;
      }

      bool first = op_queue_[op_type].enqueue_operation(descriptor, op);
      scheduler_.work_started();
      if (first)
        interrupt_unlocked();
    }

    void ionotify_reactor::cancel_ops(socket_type descriptor,
                                      ionotify_reactor::per_descriptor_data&)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      cancel_ops_unlocked(descriptor, asio::error::operation_aborted);
    }

    void
    ionotify_reactor::cancel_ops_by_key(socket_type descriptor,
                                        ionotify_reactor::per_descriptor_data&,
                                        int op_type, void* cancellation_key)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      op_queue<operation> ops;
      bool need_interrupt = op_queue_[op_type].cancel_operations_by_key(
          descriptor, ops, cancellation_key, asio::error::operation_aborted);
      scheduler_.post_deferred_completions(ops);
      if (need_interrupt)
        interrupt_unlocked();
    }

    void
    ionotify_reactor::deregister_descriptor_unlocked(socket_type descriptor)
    {
      // If run() has never seen this fd, we don't need to do anything
      if (static_cast<size_t>(descriptor) >= fdmap_.size())
        return;

      // An fd has been deregistered. If it's armed, we need to invalidate its pulses.
      fdmap_[descriptor].armed_ = 0;
      fdmap_[descriptor].ops_ = 0;
      fdmap_[descriptor].wanted_ = 0;
      if (fdmap_[descriptor].ioev_.sigev_notify != 0)
      {
        if (MsgUnregisterEvent(&fdmap_[descriptor].ioev_) == 0)
          fdmap_[descriptor].ioev_.sigev_notify = 0;
      }
    }

    void ionotify_reactor::deregister_descriptor(
        socket_type descriptor, ionotify_reactor::per_descriptor_data& /*d*/,
        bool /*cancel_ops*/)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);

      cancel_ops_unlocked(descriptor, asio::error::operation_aborted);
      deregister_descriptor_unlocked(descriptor);
    }

    void ionotify_reactor::deregister_internal_descriptor(
        socket_type descriptor, ionotify_reactor::per_descriptor_data& /*d*/)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      op_queue<operation> ops;
      for (int i = 0; i < max_ops; ++i)
        op_queue_[i].cancel_operations(descriptor, ops);
      deregister_descriptor_unlocked(descriptor);
    }

    socket_type
    ionotify_reactor::fdmap::set(reactor_op_queue<socket_type>& operations,
                                 op_queue<operation>& /*ops*/, op_types op)
    {
      socket_type maxfd = invalid_socket;
      const op_bits opbit = op_bit(op);
      reactor_op_queue<socket_type>::iterator i = operations.begin();
      while (i != operations.end())
      {
        reactor_op_queue<socket_type>::iterator op_iter = i++;
        socket_type fd = op_iter->first;
        if (maxfd == invalid_socket || fd > maxfd)
        {
          maxfd = fd;
          if (static_cast<size_t>(maxfd) >= size())
            resize(maxfd + 1);
        }
        (*this)[fd].ops_ |= opbit;
      }
      return maxfd;
    }

    void
    ionotify_reactor::fdmap::perform(reactor_op_queue<socket_type>& operations,
                                     op_queue<operation>& ops, op_types op)
    {
      const op_bits opbit = op_bit(op);
      reactor_op_queue<socket_type>::iterator i = operations.begin();
      while (i != operations.end())
      {
        reactor_op_queue<socket_type>::iterator op_iter = i++;
        socket_type fd = op_iter->first;
        if (static_cast<size_t>(fd) >= size())
          continue;

        fdstate& state = (*this)[fd];
        if (state.ops_ & opbit)
        {
          state.ops_ &= ~opbit;
          operations.perform_operations(op_iter, ops);
        }
      }
    }

    void ionotify_reactor::cleanup_descriptor_data(
        ionotify_reactor::per_descriptor_data&)
    {
    }

    void ionotify_reactor::run(long usec, op_queue<operation>& ops)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);

      // In each element of fdmap_:
      //  * ops_ is zero except when we're in this function
      //  * armed_ keep track of the fd's state between select() calls
      //  * wanted_ is used by select() but not meaningful between calls
      //  * ioev is used by select() to get SI_NOTIFY pulse when data from fd is ready

      socket_type max_fd = invalid_socket;
      bool have_work_to_do = !timer_queues_.all_empty();
      for (int i = 0; i < max_select_ops; ++i)
      {
        have_work_to_do = have_work_to_do || !op_queue_[i].empty();
        socket_type maxfd =
            fdmap_.set(op_queue_[i], ops, static_cast<op_types>(i));
        if (max_fd == invalid_socket || maxfd > max_fd)
          max_fd = maxfd;
      }

      // We can return immediately if there's no work to do and the reactor is
      // not supposed to block.
      if (!usec && !have_work_to_do)
        return;

      // Determine how long to block while waiting for events.
      long timeout_usec = 0;
      if (usec && !interrupted_)
        timeout_usec = get_timeout_usec(usec);

      // Block on the select call until descriptors become ready.
      int retval = select(lock, max_fd, timeout_usec);

      lock.lock();
      interrupted_ = false;

      // Dispatch all ready operations.
      if (retval > 0)
      {

        // Exception operations must be processed first to ensure that any
        // out-of-band data is read before normal data.
        for (int i = max_select_ops - 1; i >= 0; --i)
          fdmap_.perform(op_queue_[i], ops, static_cast<op_types>(i));
      }
      timer_queues_.get_ready_timers(ops);
    }

    void ionotify_reactor::interrupt_unlocked()
    {
      if (!interrupted_)
      {
        MsgSendPulse(coid_, PULSE_PRIO, INTERRUPT_PULSE_CODE, 0);
        interrupted_ = true;
      }
    }

    void ionotify_reactor::interrupt()
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      interrupt_unlocked();
    }

    int ionotify_reactor::select(asio::detail::mutex::scoped_lock& lock,
                                 socket_type max_fd, long timeout_usec)
    {
      static const int opmap[max_select_ops] = {
        _NOTIFY_COND_INPUT,
        _NOTIFY_COND_OUTPUT,
        _NOTIFY_COND_OBAND,
      };
      int nready = 0;   // How many fds are ready
      int narmed = 0;   // How many pulses we've armed
      int nalready = 0; // How many pulses were already armed

      if (max_fd != invalid_socket)
      {
        for (socket_type fd = 0; fd <= max_fd; ++fd)
        {
          fdstate& state = fdmap_[fd];
          if (state.ops_ == 0)
            state.wanted_ = false;
          else
          {
            state.wanted_ = true;
            // Do we already have an armed pulse that matches what we want?
            // (Note that armed for *more* than we want would require special handling if it arrives with just ops that we don't want)
            if (state.ops_ == state.armed_)
            {
              ++nalready;
              state.ops_ = 0; // We'll set them when we get a pulse
            }
            else
            {
              // A pulse is either not armed or not a match.
              // We'll need to call ionotify and ignore any old pulse.

              int io_ops = 0;
              for (int i = 0; i < max_select_ops; ++i)
              {
                op_bits opbit = op_bit(static_cast<op_types>(i));
                if (state.ops_ & opbit)
                  io_ops |= opmap[i];
              }

              sigevent* pev = &state.ioev_;
              if (pev->sigev_notify == 0)
              {
                SIGEV_PULSE_INIT(pev, coid_, PULSE_PRIO, SI_NOTIFY, fd << 1);
                SIGEV_MAKE_UPDATEABLE(pev);
                if (MsgRegisterEvent(pev, fd) < 0)
                {
                  asio::error_code ec(errno,
                                      asio::error::get_system_category());
                  asio::detail::throw_error(ec, "MsgRegisterEvent failed:");
                }
              }
              pev->sigev_value.sival_int ^= SERIAL_BIT;
              int rc = ::ionotify(fd, _NOTIFY_ACTION_POLLARM, io_ops, pev);
              if (rc != 0)
              {
                if (rc == -1)
                {
                  asio::error_code ec(errno,
                                      asio::error::get_system_category());
                  asio::detail::throw_error(ec, "ionotify failed:");
                }
                // Unset any bits that aren't ready
                for (int i = 0; i < max_select_ops; ++i)
                  if (!(rc & opmap[i]))
                    state.ops_ &= ~op_bit(static_cast<op_types>(i));
                ++nready;
                state.armed_ = 0;
              }
              else
              {
                ++narmed;
                state.armed_ = state.ops_;
                state.ops_ = 0; // We'll set them when we get a pulse
              }
            }
          }
        }
      }

      lock.unlock();

      // Now collect pulses.  If appropriate, wait for them.

      uint64_t timeout_nsec, *ntime;
      if (nready == 0 && timeout_usec != 0)
        ntime = &(timeout_nsec = timeout_usec * 1000uLL);
      else
        ntime = nullptr;

      _pulse pulse;
      int rc;

      TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, 0, ntime, 0);
      while ((rc = MsgReceivePulse_r(chid_, &pulse, sizeof(pulse), NULL)) ==
             EOK)
      {
        switch (pulse.code)
        {
          case INTERRUPT_PULSE_CODE:
            if (ntime)
              ntime = nullptr;
            break;
          case SI_NOTIFY:
          {
            socket_type fd = (pulse.value.sival_int & _NOTIFY_DATA_MASK) >> 1;
            asio::detail::mutex::scoped_lock lk(mutex_);
            if (fdmap_.is_valid(fd))
            {
              fdstate& state = fdmap_[fd];
              state.armed_ = 0;
              if ((pulse.value.sival_int & _NOTIFY_DATA_MASK) ==
                  static_cast<unsigned int>(state.ioev_.sigev_value.sival_int))
              {
                if (state.wanted_)
                {
                  for (int i = 0; i < max_select_ops; ++i)
                    if (pulse.value.sival_int & opmap[i])
                    {
                      state.ops_ |= op_bit(static_cast<op_types>(i));
                      ntime = nullptr; // We have a ready fd.  No more waiting.
                    }
                  ++nready;
                }
              }
            }
            break;
          }
          default:
            break;
        }
        TimerTimeout(CLOCK_MONOTONIC, _NTO_TIMEOUT_RECEIVE, 0, ntime, 0);
      }

      return nready;
    }

    void ionotify_reactor::do_add_timer_queue(timer_queue_base& queue)
    {
      mutex::scoped_lock lock(mutex_);
      timer_queues_.insert(&queue);
    }

    void ionotify_reactor::do_remove_timer_queue(timer_queue_base& queue)
    {
      mutex::scoped_lock lock(mutex_);
      timer_queues_.erase(&queue);
    }

    long ionotify_reactor::get_timeout_usec(long usec)
    {
      // By default we will wait no longer than 5 minutes. This will ensure that
      // any changes to the system clock are detected after no longer than this.
      const long max_usec = 5 * 60 * 1000 * 1000;
      return timer_queues_.wait_duration_usec(
          (usec < 0 || max_usec < usec) ? max_usec : usec);
    }

    void ionotify_reactor::cancel_ops_unlocked(socket_type descriptor,
                                               const asio::error_code& ec)
    {
      bool need_interrupt = false;
      op_queue<operation> ops;
      for (int i = 0; i < max_ops; ++i)
        need_interrupt = op_queue_[i].cancel_operations(descriptor, ops, ec) ||
                         need_interrupt;
      scheduler_.post_deferred_completions(ops);
      if (need_interrupt)
        interrupt_unlocked();
    }

    void ionotify_reactor::create_pulse_channel()
    {
      const unsigned channelflags = _NTO_CHF_FIXED_PRIORITY | _NTO_CHF_PRIVATE;
      const unsigned connectflags = _NTO_COF_REG_EVENTS;
      chid_ = ChannelCreate_r(channelflags);
      if (chid_ < 0)
      {
        asio::error_code ec(-chid_, asio::error::get_system_category());
        asio::detail::throw_error(ec, "ChannelCreate_r failed:");
      }
      else
      {
        coid_ = ConnectAttach(0, 0, chid_, _NTO_SIDE_CHANNEL, connectflags);
        if (coid_ < 0)
        {
          asio::error_code ec(errno, asio::error::get_system_category());
          asio::detail::throw_error(ec, "ConnectAttach failed:");
        }
      }
    }

  } // namespace detail
} // namespace asio

#include <asio/detail/pop_options.hpp>

#endif // defined(ASIO_HAS_IONOTIFY)

#endif // ASIO_DETAIL_IMPL_IONOTIFY_REACTOR_IPP
