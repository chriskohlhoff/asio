//
// service_registry.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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

#include "asio/service_factory.hpp"
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
    while (first_service_)
    {
      service_holder_base* next_service = first_service_->next_;
      delete first_service_;
      first_service_ = next_service;
    }
  }

  // Get the service object corresponding to the specified service type. Will
  // create a new service object automatically if no such object already
  // exists. Ownership of the service object is not transferred to the caller.
  template <typename Service>
  Service& get_service(service_factory<Service> factory)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // First see if there is an existing service object for the given type.
    service_holder_base* service = first_service_;
    while (service)
    {
      if (service->is_same_type(typeid(Service)))
      {
        service_holder<Service>* typed_service =
          static_cast<service_holder<Service>*>(service);
        return typed_service->service();
      }
      service = service->next_;
    }

    // Create a new service object. The service registry's mutex is not locked
    // at this time to allow for nested calls into this function from the new
    // service's constructor.
    lock.unlock();
    std::auto_ptr<service_holder<Service> > new_service(
        new service_holder<Service>(factory, owner_));
    Service& new_service_ref = new_service->service();
    lock.lock();

    // Service was successfully initialised, pass ownership to registry.
    new_service->next_ = first_service_;
    first_service_ = new_service.release();

    return new_service_ref;
  }

private:
  // The base holder for a single service.
  class service_holder_base
    : private noncopyable
  {
  public:
    // Constructor.
    service_holder_base()
      : next_(0)
    {
    }

    // Destructor.
    virtual ~service_holder_base()
    {
    }

    // Determine whether this service is the given type.
    virtual bool is_same_type(const std::type_info&) = 0;

    // A pointer to the next service holder in the list.
    service_holder_base* next_;
  };

  // Template used as the concrete holder for the service types.
  template <typename Service>
  class service_holder
    : public service_holder_base
  {
  public:
    // Constructor.
    service_holder(service_factory<Service>& factory, Owner& owner)
      : service_(factory.create(owner))
    {
    }

    // Destructor.
    virtual ~service_holder()
    {
      delete service_;
    }

    // Determine whether this service is the given type.
    virtual bool is_same_type(const std::type_info& other_info)
    {
      return other_info == typeid(Service) ? true : false;
    }

    // Get a pointer to the contained service.
    Service& service()
    {
      return *service_;
    }

  private:
    // The contained service object.
    Service* service_;
  };

  // Mutex to protect access to internal data.
  asio::detail::mutex mutex_;

  // The owner of this service registry and the services it contains.
  Owner& owner_;

  // The first service in the list of contained services.
  service_holder_base* first_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SERVICE_REGISTRY_HPP
