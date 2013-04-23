//
// handler_token.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_HANDLER_TOKEN_HPP
#define ASIO_HANDLER_TOKEN_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/handler_type.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

/// An interface for customising the behaviour of an initiating function.
/**
 * This template may be specialised for user-defined handler types.
 */
template <typename Handler>
class handler_token
{
public:
  /// The return type of the initiating function.
  typedef void type;

  /// Construct a token from a given handler.
  /**
   * When using a specalised handler_token, the constructor has an opportunity
   * to initialise some state associated with the handler, which is then
   * returned from the initiating function.
   */
  explicit handler_token(Handler&)
  {
  }

  /// Obtain the value to be returned from the initiating function.
  type get()
  {
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#define ASIO_INITFN_RESULT_TYPE(h, sig) \
  typename handler_token<typename handler_type<h, sig>::type>::type

#define ASIO_INITFN_TOKEN(h, sig) \
  handler_token<typename handler_type<h, sig>::type>

#endif // ASIO_HANDLER_TOKEN_HPP
