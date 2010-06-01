//
// detail/resolver_service_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_RESOLVER_SERVICE_BASE_HPP
#define ASIO_DETAIL_RESOLVER_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/scoped_ptr.hpp>
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/thread.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class resolver_service_base
{
public:
  // The implementation type of the resolver. A cancellation token is used to
  // indicate to the background thread that the operation has been cancelled.
  typedef socket_ops::shared_cancel_token_type implementation_type;

  // Constructor.
  resolver_service_base(asio::io_service& io_service)
    : io_service_impl_(asio::use_service<io_service_impl>(io_service)),
      work_io_service_(new asio::io_service),
      work_io_service_impl_(asio::use_service<
          io_service_impl>(*work_io_service_)),
      work_(new asio::io_service::work(*work_io_service_)),
      work_thread_(0)
  {
  }

  // Destructor.
  ~resolver_service_base()
  {
    shutdown_service();
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
    work_.reset();
    if (work_io_service_)
    {
      work_io_service_->stop();
      if (work_thread_)
      {
        work_thread_->join();
        work_thread_.reset();
      }
      work_io_service_.reset();
    }
  }

  // Construct a new resolver implementation.
  void construct(implementation_type& impl)
  {
    impl.reset(static_cast<void*>(0), socket_ops::noop_deleter());
  }

  // Destroy a resolver implementation.
  void destroy(implementation_type&)
  {
  }

  // Cancel pending asynchronous operations.
  void cancel(implementation_type& impl)
  {
    impl.reset(static_cast<void*>(0), socket_ops::noop_deleter());
  }

protected:
  // Helper function to start an asynchronous resolve operation.
  void start_resolve_op(operation* op)
  {
    start_work_thread();
    io_service_impl_.work_started();
    work_io_service_impl_.post_immediate_completion(op);
  }

  // Helper class to perform exception-safe cleanup of addrinfo objects.
  class auto_addrinfo
    : private asio::detail::noncopyable
  {
  public:
    explicit auto_addrinfo(asio::detail::addrinfo_type* ai)
      : ai_(ai)
    {
    }

    ~auto_addrinfo()
    {
      if (ai_)
        socket_ops::freeaddrinfo(ai_);
    }

    operator asio::detail::addrinfo_type*()
    {
      return ai_;
    }

  private:
    asio::detail::addrinfo_type* ai_;
  };

  // Helper class to run the work io_service in a thread.
  class work_io_service_runner
  {
  public:
    work_io_service_runner(asio::io_service& io_service)
      : io_service_(io_service) {}
    void operator()() { io_service_.run(); }
  private:
    asio::io_service& io_service_;
  };

  // Start the work thread if it's not already running.
  void start_work_thread()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (!work_thread_)
    {
      work_thread_.reset(new asio::detail::thread(
            work_io_service_runner(*work_io_service_)));
    }
  }

  // The io_service implementation used to post completions.
  io_service_impl& io_service_impl_;

private:
  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // Private io_service used for performing asynchronous host resolution.
  boost::scoped_ptr<asio::io_service> work_io_service_;

  // The work io_service implementation used to post completions.
  io_service_impl& work_io_service_impl_;

  // Work for the private io_service to perform.
  boost::scoped_ptr<asio::io_service::work> work_;

  // Thread used for running the work io_service's run loop.
  boost::scoped_ptr<asio::detail::thread> work_thread_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_RESOLVER_SERVICE_BASE_HPP
