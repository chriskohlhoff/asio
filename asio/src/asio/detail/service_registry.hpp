//
// service_registry.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
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
#include <list>
#include <map>
#include <boost/noncopyable.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/service_provider.hpp"
#include "asio/service_provider_factory.hpp"

namespace asio {
namespace detail {

class service_registry
  : private boost::noncopyable
{
public:
  // Constructor.
  service_registry(demuxer& d, service_provider_factory& s);

  // Destructor.
  ~service_registry();

  // Ask a provider to return the service interface corresponding to the
  // given type. Ownership of the service interface is not transferred to the
  // caller. Throws service_unavailable if no provider is able to provide the
  // requested service.
  service& get_service(const service_type_id& service_type);

private:
  typedef boost::shared_ptr<service_provider> service_provider_ptr;

  // Mutex to protect access to internal data.
  boost::recursive_mutex mutex_;

  // The demuxer that owns this service registry.
  demuxer& demuxer_;

  // Factory used to create new service providers.
  service_provider_factory& service_provider_factory_;

  // The type for a list of registered providers.
  typedef std::list<service_provider_ptr> provider_list;

  // The list of registered providers.
  provider_list providers_;

  // The type for a mapping from service type to service.
  typedef std::map<service_type_id, service*> type_to_service_map;

  // The mapping from service type to service.
  type_to_service_map type_to_service_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SERVICE_REGISTRY_HPP
