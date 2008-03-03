//
// noncopyable.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_NONCOPYABLE_HPP
#define ASIO_DETAIL_NONCOPYABLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <boost/noncopyable.hpp>
#include <boost/detail/workaround.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
// Redefine the noncopyable class for Borland C++ since that compiler does not
// apply the empty base optimisation unless the base class contains a dummy
// char data member.
class noncopyable
{
protected:
  noncopyable() {}
  ~noncopyable() {}
private:
  noncopyable(const noncopyable&);
  const noncopyable& operator=(const noncopyable&);
  char dummy_;
};
#else
using boost::noncopyable;
#endif

} // namespace detail

using asio::detail::noncopyable;

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_NONCOPYABLE_HPP
