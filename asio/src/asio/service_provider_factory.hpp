//
// service_provider_factory.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SERVICE_PROVIDER_FACTORY_HPP
#define ASIO_SERVICE_PROVIDER_FACTORY_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

class demuxer;
class service_provider;
class service_type_id;

class service_provider_factory
{
public:
  // Destructor.
  virtual ~service_provider_factory();

  // Create a service provider to support the given service type id. Returns 0
  // if a provider supporting the given service type cannot be created.
  virtual service_provider* create_service_provider(demuxer& owning_demuxer,
      const service_type_id& service_type) = 0;

  // Get the default service provider factory.
  static service_provider_factory& default_factory();
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SERVICE_PROVIDER_FACTORY_HPP
