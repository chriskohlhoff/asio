//
// service_registry.hpp
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

#ifndef ASIO_DETAIL_SERVICE_REGISTRY_HPP
#define ASIO_DETAIL_SERVICE_REGISTRY_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <typeinfo>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service_factory.hpp"
#include "asio/detail/mutex.hpp"

namespace asio {
namespace detail {

template <typename Owner>
class service_registry
  : private boost::noncopyable
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
        return typed_service->get_service(factory, owner_);
      }
      service = service->next_;
    }

    // We need to create a new service object.
    service_holder<Service>* new_service = new service_holder<Service>;
    new_service->next_ = first_service_;
    first_service_ = new_service;

    // Release the lock to allow calls back into get_service from the new
    // service's constructor.
    lock.unlock();

    return new_service->get_service(factory, owner_);
  }

private:
  // The base holder for a single service.
  class service_holder_base
    : private boost::noncopyable
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
    service_holder()
      : mutex_(),
        service_(0)
    {
    }

    // Destructor.
    ~service_holder()
    {
      delete service_;
    }

    // Determine whether this service is the given type.
    virtual bool is_same_type(const std::type_info& other_info)
    {
      return other_info == typeid(Service) ? true : false;
    }

    // Get a pointer to the contained service.
    Service& get_service(service_factory<Service>& factory, Owner& owner)
    {
      asio::detail::mutex::scoped_lock lock(mutex_);
      if (!service_)
        service_ = factory.create(owner);
      return *service_;
    }

  private:
    // Mutex to protect access to the contained service object.
    asio::detail::mutex mutex_;

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
