//
// default_error_handler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DEFAULT_ERROR_HANDLER_HPP
#define ASIO_DEFAULT_ERROR_HANDLER_HPP

#include "asio/detail/push_options.hpp"

namespace asio {

/// The default error handler. Always throws the error as an exception.
class default_error_handler
{
public:
  /// Evaluate the expression.
  template <typename Error>
  bool operator()(const Error& err)
  {
    if (err)
      throw err;
    return false;
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DEFAULT_ERROR_HANDLER_HPP
