//
// default_error_handler.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
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
