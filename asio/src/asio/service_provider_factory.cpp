//
// service_provider_factory.cpp
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

#include "asio/service_provider_factory.hpp"
#include "asio/detail/default_service_provider_factory.hpp"

namespace asio {

service_provider_factory::
~service_provider_factory()
{
}

namespace { detail::default_service_provider_factory the_default_factory; }

service_provider_factory&
service_provider_factory::
default_factory()
{
  return the_default_factory;
}

} // namespace asio
