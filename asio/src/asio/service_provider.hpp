//
// service_provider.hpp
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

#ifndef ASIO_SERVICE_PROVIDER_HPP
#define ASIO_SERVICE_PROVIDER_HPP

#include "asio/detail/push_options.hpp"

#include "asio/service_type_id.hpp"

namespace asio {

class service;

/// The service_provider class is a base class for all implementations of
/// asynchronous functionality, and that provide one or more service
/// interfaces.
class service_provider
{
public:
  /// Destructor.
  virtual ~service_provider();

  /// Ask the provider to return the service interface corresponding to the
  /// given type. Ownership of the service interface is not transferred to the
  /// caller. Returns 0 if the provider does not support the requested service.
  service* get_service(const service_type_id& service_type);

private:
  /// Return the service interface corresponding to the given type.
  virtual service* do_get_service(const service_type_id& service_type) = 0;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SERVICE_PROVIDER_HPP
