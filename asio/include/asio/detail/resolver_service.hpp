//
// detail/resolver_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_RESOLVER_SERVICE_HPP
#define ASIO_DETAIL_RESOLVER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <boost/scoped_ptr.hpp>
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/ip/basic_resolver_iterator.hpp"
#include "asio/ip/basic_resolver_query.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/resolve_endpoint_op.hpp"
#include "asio/detail/resolve_op.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/thread.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class resolver_service
  : public asio::detail::service_base<resolver_service<Protocol> >
{
private:
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

public:
  // The implementation type of the resolver. A cancellation token is used to
  // indicate to the background thread that the operation has been cancelled.
  typedef socket_ops::shared_cancel_token_type implementation_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  // The query type.
  typedef asio::ip::basic_resolver_query<Protocol> query_type;

  // The iterator type.
  typedef asio::ip::basic_resolver_iterator<Protocol> iterator_type;

  // Constructor.
  resolver_service(asio::io_service& io_service)
    : asio::detail::service_base<
        resolver_service<Protocol> >(io_service),
      mutex_(),
      io_service_impl_(asio::use_service<io_service_impl>(io_service)),
      work_io_service_(new asio::io_service),
      work_io_service_impl_(asio::use_service<
          io_service_impl>(*work_io_service_)),
      work_(new asio::io_service::work(*work_io_service_)),
      work_thread_(0)
  {
  }

  // Destructor.
  ~resolver_service()
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

  // Resolve a query to a list of entries.
  iterator_type resolve(implementation_type&, const query_type& query,
      asio::error_code& ec)
  {
    asio::detail::addrinfo_type* address_info = 0;

    socket_ops::getaddrinfo(query.host_name().c_str(),
        query.service_name().c_str(), query.hints(), &address_info, ec);
    auto_addrinfo auto_address_info(address_info);

    return ec ? iterator_type() : iterator_type::create(
        address_info, query.host_name(), query.service_name());
  }

  // Asynchronously resolve a query to a list of entries.
  template <typename Handler>
  void async_resolve(implementation_type& impl, const query_type& query,
      Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef resolve_op<Protocol, Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl, query, io_service_impl_, handler);

    if (work_io_service_)
    {
      start_work_thread();
      io_service_impl_.work_started();
      work_io_service_impl_.post_immediate_completion(p.p);
      p.v = p.p = 0;
    }
  }

  // Resolve an endpoint to a list of entries.
  iterator_type resolve(implementation_type&,
      const endpoint_type& endpoint, asio::error_code& ec)
  {
    char host_name[NI_MAXHOST];
    char service_name[NI_MAXSERV];
    socket_ops::sync_getnameinfo(endpoint.data(), endpoint.size(),
        host_name, NI_MAXHOST, service_name, NI_MAXSERV,
        endpoint.protocol().type(), ec);

    return ec ? iterator_type() : iterator_type::create(
        endpoint, host_name, service_name);
  }

  // Asynchronously resolve an endpoint to a list of entries.
  template <typename Handler>
  void async_resolve(implementation_type& impl, const endpoint_type& endpoint,
      Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef resolve_endpoint_op<Protocol, Handler> op;
    typename op::ptr p = { boost::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(impl, endpoint, io_service_impl_, handler);

    if (work_io_service_)
    {
      start_work_thread();
      io_service_impl_.work_started();
      work_io_service_impl_.post_immediate_completion(p.p);
      p.v = p.p = 0;
    }
  }

private:
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

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The io_service implementation used to post completions.
  io_service_impl& io_service_impl_;

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

#endif // ASIO_DETAIL_RESOLVER_SERVICE_HPP
