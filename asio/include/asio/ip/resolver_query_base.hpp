//
// resolver_query_base.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_IP_RESOLVER_QUERY_BASE_HPP
#define ASIO_IP_RESOLVER_QUERY_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/socket_types.hpp"

namespace asio {
namespace ip {

/// The resolver_query_base class is used as a base for the
/// basic_resolver_query class templates to provide a common place to define
/// the flag constants.
class resolver_query_base
{
public:
#if defined(GENERATING_DOCUMENTATION)
  /// Determine the canonical name of the host specified in the query.
  static const int canonical_name = implementation_defined;

  /// Host name should be treated as a numeric string defining an IPv4 or IPv6
  /// address and no name resolution should be attempted.
  static const int numeric_host = implementation_defined;

  /// Indicate that returned endpoint is intended for use as a locally bound
  /// socket endpoint.
  static const int passive = implementation_defined;
#else
  BOOST_STATIC_CONSTANT(int, canonical_name = AI_CANONNAME);
  BOOST_STATIC_CONSTANT(int, numeric_host = AI_NUMERICHOST);
  BOOST_STATIC_CONSTANT(int, passive = AI_PASSIVE);
#endif

protected:
  /// Protected destructor to prevent deletion through this type.
  ~resolver_query_base()
  {
  }

#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
private:
  // Workaround to enable the empty base optimisation with Borland C++.
  char dummy_;
#endif
};

} // namespace ip
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_IP_RESOLVER_QUERY_BASE_HPP
