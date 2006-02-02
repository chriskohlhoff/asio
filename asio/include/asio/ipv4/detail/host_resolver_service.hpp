//
// host_resolver_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IPV4_DETAIL_HOST_RESOLVER_SERVICE_HPP
#define ASIO_IPV4_DETAIL_HOST_RESOLVER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cstring>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/thread.hpp"
#include "asio/ipv4/host.hpp"

namespace asio {
namespace ipv4 {
namespace detail {

template <typename IO_Service>
class host_resolver_service
  : private boost::noncopyable
{
private:
  // Helper class to perform exception-safe cleanup of hostent objects.
  class auto_hostent
    : private boost::noncopyable
  {
  public:
    explicit auto_hostent(hostent* h)
      : h_(h)
    {
    }

    ~auto_hostent()
    {
      if (h_)
        asio::detail::socket_ops::freehostent(h_);
    }

    operator hostent*()
    {
      return h_;
    }

  private:
    hostent* h_;
  };

public:
  // The implementation type of the host resolver. The shared pointer is used
  // as a cancellation token to indicate to the background thread that the
  // operation has been cancelled.
  typedef boost::shared_ptr<void> implementation_type;

  // Constructor.
  host_resolver_service(IO_Service& d)
    : mutex_(),
      io_service_(d),
      work_io_service_(),
      work_(new typename IO_Service::work(work_io_service_)),
      work_thread_(0)
  {
  }

  // Destructor.
  ~host_resolver_service()
  {
    delete work_;
    if (work_thread_)
    {
      work_thread_->join();
      delete work_thread_;
    }
  }

  // The io_service type for this service.
  typedef IO_Service io_service_type;

  // Get the io_service associated with the service.
  io_service_type& io_service()
  {
    return io_service_;
  }

  // Construct a new host resolver implementation.
  void construct(implementation_type& impl)
  {
  }

  // Destroy a host resolver implementation.
  void destroy(implementation_type& impl)
  {
  }

  /// Cancel pending asynchronous operations.
  void cancel(implementation_type& impl)
  {
    impl.reset(0);
  }

  // Get host information for the local machine.
  template <typename Error_Handler>
  void local(implementation_type& impl, host& h, Error_Handler error_handler)
  {
    char name[1024];
    if (asio::detail::socket_ops::gethostname(name, sizeof(name)) != 0)
    {
      error_handler(asio::error(
            asio::detail::socket_ops::get_error()));
    }
    else
    {
      by_name(impl, h, name, error_handler);
    }
  }

  // Get host information for a specified address.
  template <typename Error_Handler>
  void by_address(implementation_type& impl, host& h, const address& addr,
      Error_Handler error_handler)
  {
    hostent ent;
    char buf[8192] = ""; // Size recommended by Stevens, UNPv1.
    int error = 0;
    in_addr a;
    a.s_addr = asio::detail::socket_ops::host_to_network_long(
        addr.to_ulong());
    auto_hostent result(asio::detail::socket_ops::gethostbyaddr(
          reinterpret_cast<const char*>(&a), sizeof(in_addr), AF_INET, &ent,
          buf, sizeof(buf), &error));
    if (result == 0)
      error_handler(asio::error(error));
    else if (ent.h_length != sizeof(in_addr))
      error_handler(asio::error(asio::error::host_not_found));
    else
      populate_host_object(h, ent);
  }

  template <typename Handler>
  class by_address_handler
  {
  public:
    by_address_handler(implementation_type impl, host& h, const address& addr,
        IO_Service& io_service, Handler handler)
      : impl_(impl),
        host_(h),
        address_(addr),
        io_service_(io_service),
        work_(io_service),
        handler_(handler)
    {
    }

    void operator()()
    {
      // Check if the operation has been cancelled.
      if (impl_.expired())
      {
        io_service_.post(asio::detail::bind_handler(handler_,
              asio::error(asio::error::operation_aborted)));
        return;
      }

      // Perform the blocking host resolution operation.
      hostent ent;
      char buf[8192] = ""; // Size recommended by Stevens, UNPv1.
      int error = 0;
      in_addr a;
      a.s_addr = asio::detail::socket_ops::host_to_network_long(
          address_.to_ulong());
      auto_hostent result(asio::detail::socket_ops::gethostbyaddr(
            reinterpret_cast<const char*>(&a), sizeof(in_addr), AF_INET, &ent,
            buf, sizeof(buf), &error));
      asio::error e(asio::error::success);
      if (result == 0)
        e = asio::error(error);
      else if (ent.h_length != sizeof(in_addr))
        e = asio::error(asio::error::host_not_found);
      else
        populate_host_object(host_, ent);
      io_service_.post(asio::detail::bind_handler(handler_, e));
    }

  private:
    boost::weak_ptr<void> impl_;
    host& host_;
    address address_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Handler handler_;
  };

  // Asynchronously get host information for a specified address.
  template <typename Handler>
  void async_by_address(implementation_type& impl, host& h, const address& addr,
      Handler handler)
  {
    start_work_thread();
    work_io_service_.post(
        by_address_handler<Handler>(
          impl, h, addr, io_service_, handler));
  }

  // Get host information for a named host.
  template <typename Error_Handler>
  void by_name(implementation_type& impl, host& h, const std::string& name,
      Error_Handler error_handler)
  {
    hostent ent;
    char buf[8192] = ""; // Size recommended by Stevens, UNPv1.
    int error = 0;
    auto_hostent result(asio::detail::socket_ops::gethostbyname(
          name.c_str(), &ent, buf, sizeof(buf), &error));
    if (result == 0)
      error_handler(asio::error(error));
    else if (ent.h_addrtype != AF_INET || ent.h_length != sizeof(in_addr))
      error_handler(asio::error(asio::error::host_not_found));
    else
      populate_host_object(h, ent);
  }

  template <typename Handler>
  class by_name_handler
  {
  public:
    by_name_handler(implementation_type impl, host& h, const std::string& name,
        IO_Service& io_service, Handler handler)
      : impl_(impl),
        host_(h),
        name_(name),
        io_service_(io_service),
        work_(io_service),
        handler_(handler)
    {
    }

    void operator()()
    {
      // Check if the operation has been cancelled.
      if (impl_.expired())
      {
        io_service_.post(asio::detail::bind_handler(handler_,
              asio::error(asio::error::operation_aborted)));
        return;
      }

      // Perform the blocking host resolution operation.
      hostent ent;
      char buf[8192] = ""; // Size recommended by Stevens, UNPv1.
      int error = 0;
      auto_hostent result(asio::detail::socket_ops::gethostbyname(
            name_.c_str(), &ent, buf, sizeof(buf), &error));
      asio::error e(asio::error::success);
      if (result == 0)
        e = asio::error(error);
      else if (ent.h_addrtype != AF_INET || ent.h_length != sizeof(in_addr))
        e = asio::error(asio::error::host_not_found);
      else
        populate_host_object(host_, ent);
      io_service_.post(asio::detail::bind_handler(handler_, e));
    }

  private:
    boost::weak_ptr<void> impl_;
    host& host_;
    std::string name_;
    IO_Service& io_service_;
    typename IO_Service::work work_;
    Handler handler_;
  };

  // Asynchronously get host information for a named host.
  template <typename Handler>
  void async_by_name(implementation_type& impl, host& h,
      const std::string& name, Handler handler)
  {
    start_work_thread();
    work_io_service_.post(
        by_name_handler<Handler>(
          impl, h, name, io_service_, handler));
  }

  // Populate a host object from a hostent structure.
  static void populate_host_object(host& h, hostent& ent)
  {
    std::vector<std::string> aliases;
    for (char** alias = ent.h_aliases; *alias; ++alias)
      aliases.push_back(*alias);

    std::vector<address> addresses;
    for (char** addr = ent.h_addr_list; *addr; ++addr)
    {
      using namespace std; // For memcpy.
      in_addr a;
      memcpy(&a, *addr, sizeof(in_addr));
      addresses.push_back(address(
            asio::detail::socket_ops::network_to_host_long(a.s_addr)));
    }

    host tmp(ent.h_name, addresses.front(), aliases.begin(), aliases.end(),
        addresses.begin() + 1, addresses.end());
    h.swap(tmp);
  }

private:
  // Helper class to run the work io_service in a thread.
  class work_io_service_runner
  {
  public:
    work_io_service_runner(IO_Service& io_service) : io_service_(io_service) {}
    void operator()() { io_service_.run(); }
  private:
    IO_Service& io_service_;
  };

  // Start the work thread if it's not already running.
  void start_work_thread()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);
    if (work_thread_ == 0)
    {
      work_thread_ = new asio::detail::thread(
          work_io_service_runner(work_io_service_));
    }
  }

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The io_service used for dispatching handlers.
  IO_Service& io_service_;

  // Private io_service used for performing asynchronous host resolution.
  IO_Service work_io_service_;

  // Work for the private io_service to perform.
  typename IO_Service::work* work_;

  // Thread used for running the work io_service's run loop.
  asio::detail::thread* work_thread_;
};

} // namespace detail
} // namespace ipv4
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IPV4_DETAIL_HOST_RESOLVER_SERVICE_HPP
