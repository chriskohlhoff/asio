//
// service_base.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_SERVICE_BASE_HPP
#define ASIO_DETAIL_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/io_service.hpp"
#include "asio/detail/service_id.hpp"

namespace asio {
namespace detail {

// Special service base class to keep classes header-file only.
template <typename Type>
class service_base
  : public asio::io_service::service
{
public:
  static asio::detail::service_id<Type> id;

  // Constructor.
  service_base(asio::io_service& io_service)
    : asio::io_service::service(io_service)
  {
  }
};

template <typename Type>
asio::detail::service_id<Type> service_base<Type>::id;

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_SERVICE_BASE_HPP
