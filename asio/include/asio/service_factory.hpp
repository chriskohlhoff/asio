//
// service_factory.hpp
// ~~~~~~~~~~~~~~~~~~~
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

#ifndef ASIO_SERVICE_FACTORY_HPP
#define ASIO_SERVICE_FACTORY_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

/// This class may be specialised to provide custom service creation.
template <typename Service>
class service_factory
{
public:
  /// Create a service with the specified owner.
  template <typename Owner>
  Service* create(Owner& owner)
  {
    return new Service(owner);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SERVICE_FACTORY_HPP
