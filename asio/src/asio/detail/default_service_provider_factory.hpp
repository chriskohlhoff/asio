//
// default_service_provider_factory.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_DETAIL_DEFAULT_SERVICE_PROVIDER_FACTORY_HPP
#define ASIO_DETAIL_DEFAULT_SERVICE_PROVIDER_FACTORY_HPP

#include "asio/service_provider_factory.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class default_service_provider_factory
  : public service_provider_factory
{
public:
  // Create a service provider to support the given service type id. Returns 0
  // if a provider supporting the given service type cannot be created.
  virtual service_provider* create_service_provider(demuxer& owning_demuxer,
      const service_type_id& service_type);
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_DEFAULT_SERVICE_PROVIDER_FACTORY_HPP
