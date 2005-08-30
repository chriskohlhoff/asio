//
// null_error_handler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_NULL_ERROR_HANDLER_HPP
#define ASIO_NULL_ERROR_HANDLER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

namespace asio {

/// The null error handler. Always ignores the error.
class null_error_handler
{
public:
  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    return true;
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_NULL_ERROR_HANDLER_HPP
