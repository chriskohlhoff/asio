//
// service_factory.hpp
// ~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
