//
// throw_error.hpp
// ~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_THROW_ERROR_HPP
#define ASIO_DETAIL_THROW_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/error_code.hpp"
#include "asio/system_error.hpp"

namespace asio {
namespace detail {

inline void throw_error(const asio::error_code& err)
{
  if (err)
  {
    asio::system_error e(err);
    boost::throw_exception(e);
  }
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_THROW_ERROR_HPP
