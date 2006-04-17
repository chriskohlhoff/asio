//
// service_registry.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SERVICE_REGISTRY_HPP
#define ASIO_DETAIL_SERVICE_REGISTRY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <memory>
#include <typeinfo>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"

namespace asio {
namespace detail {

template <typename Owner>
class service_registry
  : private noncopyable
{
public:
  // Constructor.
  service_registry(Owner& o)
    : owner_(o),
      first_service_(0)
  {
  }

  // Destructor.
  ~service_registry()
  {
    // Shutdown all services. This must be done in a separate loop before the
    // services are destroyed since the destructors of user-defined handler
    // objects may try to access other service objects.
    typename Owner::service* service = first_service_;
    while (service)
    {
      service->shutdown_service();
      service = service->next_;
    }

    // Destroy all services.
    while (first_service_)
    {
      typename Owner::service* next_service = first_service_->next_;
      delete first_service_;
      first_service_ = next_service;
    }
  }

  // Get the service object corresponding to the specified service type. Will
  // create a new service object automatically if no such object already
  // exists. Ownership of the service object is not transferred to the caller.
  template <typename Service>
  Service& use_service()
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // First see if there is an existing service object for the given type.
    typename Owner::service* service = first_service_;
    while (service)
    {
      if (*service->type_info_ == typeid(Service))
        return *static_cast<Service*>(service);
      service = service->next_;
    }

    // Create a new service object. The service registry's mutex is not locked
    // at this time to allow for nested calls into this function from the new
    // service's constructor.
    lock.unlock();
    std::auto_ptr<Service> new_service(new Service(owner_));
    new_service->type_info_ = &typeid(Service);
    Service& new_service_ref = *new_service;
    lock.lock();

    // Check that nobody else created another service object of the same type
    // while the lock was released.
    service = first_service_;
    while (service)
    {
      if (*service->type_info_ == typeid(Service))
        return *static_cast<Service*>(service);
      service = service->next_;
    }

    // Service was successfully initialised, pass ownership to registry.
    new_service->next_ = first_service_;
    first_service_ = new_service.release();

    return new_service_ref;
  }

  // Add a service object. Returns false on error, in which case ownership of
  // the object is retained by the caller.
  template <typename Service>
  bool add_service(Service* new_service)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Check if there is an existing service object for the given type.
    typename Owner::service* service = first_service_;
    while (service)
    {
      if (*service->type_info_ == typeid(Service))
        return false;
      service = service->next_;
    }

    // Take ownership of the service object.
    new_service->type_info_ = &typeid(Service);
    new_service->next_ = first_service_;
    first_service_ = new_service;
  }

  // Check whether a service object of the specified type already exists.
  template <typename Service>
  bool has_service() const
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    typename Owner::service* service = first_service_;
    while (service)
    {
      if (*service->type_info_ == typeid(Service))
        return true;
      service = service->next_;
    }

    return false;
  }

private:
  // Mutex to protect access to internal data.
  mutable asio::detail::mutex mutex_;

  // The owner of this service registry and the services it contains.
  Owner& owner_;

  // The first service in the list of contained services.
  typename Owner::service* first_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SERVICE_REGISTRY_HPP
