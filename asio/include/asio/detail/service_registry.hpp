//
// service_registry.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
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
#include <typeinfo>
#include "asio/detail/pop_options.hpp"

#include "asio/io_service.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/service_id.hpp"

#if defined(BOOST_NO_TYPEID)
# if !defined(ASIO_NO_TYPEID)
#  define ASIO_NO_TYPEID
# endif // !defined(ASIO_NO_TYPEID)
#endif // defined(BOOST_NO_TYPEID)

namespace asio {
namespace detail {

#if defined(__GNUC__)
# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  pragma GCC visibility push (default)
# endif // (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#endif // defined(__GNUC__)

template <typename T>
class typeid_wrapper {};

#if defined(__GNUC__)
# if (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#  pragma GCC visibility pop
# endif // (__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)
#endif // defined(__GNUC__)

class service_registry
  : private noncopyable
{
public:
  // Constructor.
  service_registry(asio::io_service& o)
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
    asio::io_service::service* service = first_service_;
    while (service)
    {
      service->shutdown_service();
      service = service->next_;
    }

    // Destroy all services.
    while (first_service_)
    {
      asio::io_service::service* next_service = first_service_->next_;
      destroy(first_service_);
      first_service_ = next_service;
    }
  }

  // Get the service object corresponding to the specified service type. Will
  // create a new service object automatically if no such object already
  // exists. Ownership of the service object is not transferred to the caller.
  template <typename Service>
  Service& use_service()
  {
    asio::io_service::service::key key;
    init_key(key, Service::id);
    factory_type factory = &service_registry::create<Service>;
    return *static_cast<Service*>(do_use_service(key, factory));
  }

  // Add a service object. Returns false on error, in which case ownership of
  // the object is retained by the caller.
  template <typename Service>
  bool add_service(Service* new_service)
  {
    asio::io_service::service::key key;
    init_key(key, Service::id);
    return do_add_service(key, new_service);
  }

  // Check whether a service object of the specified type already exists.
  template <typename Service>
  bool has_service() const
  {
    asio::io_service::service::key key;
    init_key(key, Service::id);
    return do_has_service(key);
  }

private:
  // Initialise a service's key based on its id.
  void init_key(asio::io_service::service::key& key,
      const asio::io_service::id& id)
  {
    key.type_info_ = 0;
    key.id_ = &id;
  }

#if !defined(ASIO_NO_TYPEID)
  // Initialise a service's key based on its id.
  template <typename Service>
  void init_key(asio::io_service::service::key& key,
      const asio::detail::service_id<Service>& /*id*/)
  {
    key.type_info_ = &typeid(typeid_wrapper<Service>);
    key.id_ = 0;
  }
#endif // !defined(ASIO_NO_TYPEID)

  // Check if a service matches the given id.
  static bool keys_match(
      const asio::io_service::service::key& key1,
      const asio::io_service::service::key& key2)
  {
    if (key1.id_ && key2.id_)
      if (key1.id_ == key2.id_)
        return true;
    if (key1.type_info_ && key2.type_info_)
      if (*key1.type_info_ == *key2.type_info_)
        return true;
    return false;
  }

  // The type of a factory function used for creating a service instance.
  typedef asio::io_service::service*
    (*factory_type)(asio::io_service&);

  // Factory function for creating a service instance.
  template <typename Service>
  static asio::io_service::service* create(
      asio::io_service& owner)
  {
    return new Service(owner);
  }

  // Destroy a service instance.
  static void destroy(asio::io_service::service* service)
  {
    delete service;
  }

  // Helper class to manage service pointers.
  struct auto_service_ptr
  {
    asio::io_service::service* ptr_;
    ~auto_service_ptr() { destroy(ptr_); }
  };

  // Get the service object corresponding to the specified service key. Will
  // create a new service object automatically if no such object already
  // exists. Ownership of the service object is not transferred to the caller.
  asio::io_service::service* do_use_service(
      const asio::io_service::service::key& key,
      factory_type factory)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // First see if there is an existing service object with the given key.
    asio::io_service::service* service = first_service_;
    while (service)
    {
      if (keys_match(service->key_, key))
        return service;
      service = service->next_;
    }

    // Create a new service object. The service registry's mutex is not locked
    // at this time to allow for nested calls into this function from the new
    // service's constructor.
    lock.unlock();
    auto_service_ptr new_service = { factory(owner_) };
    new_service.ptr_->key_ = key;
    lock.lock();

    // Check that nobody else created another service object of the same type
    // while the lock was released.
    service = first_service_;
    while (service)
    {
      if (keys_match(service->key_, key))
        return service;
      service = service->next_;
    }

    // Service was successfully initialised, pass ownership to registry.
    new_service.ptr_->next_ = first_service_;
    first_service_ = new_service.ptr_;
    new_service.ptr_ = 0;
    return first_service_;
  }

  // Add a service object. Returns false on error, in which case ownership of
  // the object is retained by the caller.
  bool do_add_service(
      const asio::io_service::service::key& key,
      asio::io_service::service* new_service)
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    // Check if there is an existing service object with the given key.
    asio::io_service::service* service = first_service_;
    while (service)
    {
      if (keys_match(service->key_, key))
        return false;
      service = service->next_;
    }

    // Take ownership of the service object.
    new_service->key_ = key;
    new_service->next_ = first_service_;
    first_service_ = new_service;

    return true;
  }

  // Check whether a service object with the specified key already exists.
  bool do_has_service(const asio::io_service::service::key& key) const
  {
    asio::detail::mutex::scoped_lock lock(mutex_);

    asio::io_service::service* service = first_service_;
    while (service)
    {
      if (keys_match(service->key_, key))
        return true;
      service = service->next_;
    }

    return false;
  }

  // Mutex to protect access to internal data.
  mutable asio::detail::mutex mutex_;

  // The owner of this service registry and the services it contains.
  asio::io_service& owner_;

  // The first service in the list of contained services.
  asio::io_service::service* first_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SERVICE_REGISTRY_HPP
