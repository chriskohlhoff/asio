//
// service_registry.cpp
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

#include "asio/detail/service_registry.hpp"
#include <boost/throw_exception.hpp>
#include "asio/service_unavailable.hpp"

namespace asio {
namespace detail {

service_registry::
service_registry(
    demuxer& d,
    service_provider_factory& s)
  : mutex_(),
    demuxer_(d),
    service_provider_factory_(s),
    providers_(),
    type_to_service_()
{
}

service_registry::
~service_registry()
{
}

service&
service_registry::
get_service(
    const service_type_id& type)
{
  boost::recursive_mutex::scoped_lock lock(mutex_);

  type_to_service_map::iterator service_iter = type_to_service_.find(type);
  if (service_iter != type_to_service_.end())
    return *service_iter->second;

  provider_list::iterator provider_iter = providers_.begin();
  while (provider_iter != providers_.end())
  {
    if (service* the_service = (*provider_iter)->get_service(type))
    {
      type_to_service_[type] = the_service;
      return *the_service;
    }
    ++provider_iter;
  }

  service_provider_ptr provider_ptr(
      service_provider_factory_.create_service_provider(demuxer_, type));
  if (!provider_ptr)
    boost::throw_exception(service_unavailable(type));
  providers_.push_back(provider_ptr);

  service* the_service = provider_ptr->get_service(type);
  if (!the_service)
    boost::throw_exception(service_unavailable(type));
  type_to_service_[type] = the_service;
  return *the_service;
}

} // namespace detail
} // namespace asio
